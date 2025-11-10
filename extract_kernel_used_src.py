#!/usr/bin/env python3
"""
extract_kernel_used_src.py

python extract_kernel_used_src.py --src ./linux/linux-4.19.125/ --build-dir ../build/out/AX615_nand_uclibc/objs/kernel/linux/linux-4.19.125/ --out out

Extract the actual source (.c/.S) and header (.h/.hpp/...) files used in a Linux kernel build,
copy them into a target directory preserving kernel paths, and produce dependency artifacts.

Usage:
    python3 extract_kernel_used_src.py --src /path/to/linux \
        --out /path/to/output_dir \
        [--build-dir /path/to/build_output] \
        [--top-n 60] \
        [--recursive-headers] \
        [--max-search 500] \
        [--verbose]

Notes:
 - If the build was done with an output directory (make O=...), point --build-dir to that build dir
   (it contains the .cmd files). Otherwise .cmd files are searched under --src.
 - Recursive header resolution will parse header files for #include "..." and #include <...> and
   attempt to resolve those included files as well.
 - The script prefers available libraries:
     - tqdm for progress bars (optional)
     - networkx for graph creation (optional)
     - pandas for nicer stats output (optional)
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from collections import defaultdict, Counter, deque
from typing import Dict, Optional, Set, List, Tuple

# Optional libs
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

import matplotlib.pyplot as plt

# -----------------------
# Helpers & parsing
# -----------------------

TOKEN_RE = re.compile(r'([^\\\s]+?\.(?:c|C|S|s|h|hpp|hh|inc))', re.IGNORECASE)
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')

def find_cmd_files(src_root: Path) -> List[Path]:
    """Find all .cmd files under src_root."""
    return list(src_root.rglob("*.cmd"))

def extract_tokens_from_cmd(path: Path) -> Set[str]:
    """Extract tokens like 'foo.c' or 'include/linux/foo.h' from a .cmd file's text."""
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return set()
    tokens = set(m.group(1).rstrip('\\') for m in TOKEN_RE.finditer(txt))
    # normalize windows separators if any
    tokens = {t.replace('\\', '/') for t in tokens}
    return tokens

def read_file_lines(path: Path) -> List[str]:
    try:
        return path.read_text(errors="ignore").splitlines()
    except Exception:
        return []

# -----------------------
# Resolution strategy
# -----------------------

class Resolver:
    """
    Resolve token (relative path or basename) to actual file under kernel src.
    Prioritized search list:
      - absolute path (if given)
      - src_root / token (relative)
      - src_root / include / token
      - src_root / include/uapi / token
      - src_root / include/generated / token
      - arch/*/include variants
      - src_root / token's dirname variants
      - final limited walk by basename (bounded by max_search)
    """
    def __init__(self, src_root: Path, max_search: int = 500, verbose: bool = False):
        self.src_root = src_root
        self.cache: Dict[str, Optional[Path]] = {}
        self.max_search = max_search
        self.verbose = verbose
        # precompute arch include dirs
        self.arch_include_dirs = [p for p in src_root.glob("arch/*/include") if p.is_dir()]

    def resolve(self, token: str) -> Optional[Path]:
        """Resolve token -> Path or None. Token may be like 'include/linux/kernel.h' or 'kernel.h'."""
        if token in self.cache:
            return self.cache[token]

        t = token
        # absolute?
        if os.path.isabs(t):
            p = Path(t)
            if p.exists():
                self.cache[token] = p
                return p
            else:
                self.cache[token] = None
                return None

        # direct relative under src_root
        p = self.src_root / t
        if p.exists():
            self.cache[token] = p
            return p

        base = os.path.basename(t)

        # try common candidates
        trials = [
            self.src_root / t,
            self.src_root / "include" / t,
            self.src_root / "include" / "uapi" / t,
            self.src_root / "include" / "generated" / t,
            self.src_root / "include" / base,
            self.src_root / "include" / "uapi" / base,
            self.src_root / base,
        ]
        for q in trials:
            if q.exists():
                self.cache[token] = q
                return q

        # try arch include dirs
        for archd in self.arch_include_dirs:
            q = archd / t
            if q.exists():
                self.cache[token] = q
                return q
            q2 = archd / base
            if q2.exists():
                self.cache[token] = q2
                return q2

        # try mapping if token looks like 'linux/foo.h' => include/linux/foo.h
        if '/' in t:
            maybe = self.src_root / "include" / t
            if maybe.exists():
                self.cache[token] = maybe
                return maybe
            maybe = self.src_root / "include" / "uapi" / t
            if maybe.exists():
                self.cache[token] = maybe
                return maybe

        # as last resort, limited basename walk
        matches = []
        count = 0
        # Walk but avoid scanning generated/build output dirs (heuristic)
        for root, dirs, files in os.walk(self.src_root):
            # skip common heavy dirs (optional)
            if 'node_modules' in root or '.git' in root:
                continue
            if base in files:
                matches.append(Path(root) / base)
                count += 1
                if count >= self.max_search:
                    break
        if matches:
            self.cache[token] = matches[0]
            return matches[0]

        self.cache[token] = None
        return None

# -----------------------
# Header recursive parsing
# -----------------------

def parse_includes_from_header(header_path: Path) -> List[Tuple[str,str]]:
    """
    Parse #include lines from a header file.
    Return list of tuples (delimiter, included_path) where delimiter is '<' or '"'.
    """
    lines = read_file_lines(header_path)
    out = []
    for ln in lines:
        m = INCLUDE_RE.match(ln)
        if m:
            delim, included = m.group(1), m.group(2).strip()
            out.append((delim, included))
    return out

# -----------------------
# Main extraction logic
# -----------------------

def gather_tokens(cmd_paths: List[Path], use_progress: bool = True) -> Set[str]:
    tokens = set()
    it = cmd_paths
    if use_progress and HAS_TQDM:
        it = tqdm(cmd_paths, desc="Scanning .cmd files", unit="file")
    for p in it:
        tokens.update(extract_tokens_from_cmd(p))
    return tokens

def resolve_tokens(tokens: Set[str], resolver: Resolver, use_progress: bool = True) -> Tuple[List[Path], List[Path], List[str]]:
    """Return (resolved_sources, resolved_headers, unresolved_tokens)."""
    srcs = []
    hdrs = []
    unresolved = []
    tokens_list = sorted(tokens)
    it = tokens_list
    if use_progress and HAS_TQDM:
        it = tqdm(tokens_list, desc="Resolving tokens", unit="tok")
    for tok in it:
        p = resolver.resolve(tok)
        if p:
            if p.suffix.lower() in (".c", ".s", ".S", ".C"):
                srcs.append(p)
            else:
                hdrs.append(p)
        else:
            unresolved.append(tok)
    # deduplicate and sort
    srcs = sorted(set(srcs))
    hdrs = sorted(set(hdrs))
    unresolved = sorted(set(unresolved))
    return srcs, hdrs, unresolved

def build_src_to_headers_map(cmd_paths: List[Path], resolver: Resolver, use_progress: bool = True) -> Dict[Path, Set[Path]]:
    """
    For each .cmd file, attribute headers to source(s) referenced in that .cmd.
    Returns mapping Path(source) -> set(Path(headers)).
    """
    mapping: Dict[Path, Set[Path]] = defaultdict(set)
    it = cmd_paths
    if use_progress and HAS_TQDM:
        it = tqdm(cmd_paths, desc="Attributing headers to sources", unit="file")
    for cf in it:
        text = cf.read_text(errors="ignore")
        tokens_here = set(m.group(1).rstrip('\\') for m in TOKEN_RE.finditer(text))
        resolved_here = []
        for tok in tokens_here:
            rp = resolver.resolve(tok)
            if rp:
                resolved_here.append(rp)
        srcs_here = [p for p in resolved_here if p.suffix.lower() in (".c", ".s", ".S", ".C")]
        if not srcs_here:
            # try infer from .cmd filename: .foo.o.cmd -> foo.c
            b = cf.name
            m = re.match(r'\.(.+)\.o\.cmd$', b)
            if m:
                cand = cf.parent / (m.group(1) + ".c")
                if cand.exists():
                    srcs_here = [cand]
        hdrs_here = [p for p in resolved_here if p.suffix.lower() in (".h", ".hpp", ".hh", ".inc")]
        for s in srcs_here:
            mapping[s].update(hdrs_here)
    return mapping

def recursive_header_resolution(initial_headers: Set[Path], resolver: Resolver, max_recursive: int = 10000, use_progress: bool = True) -> Set[Path]:
    """
    Given a set of headers, parse and resolve headers they include and so on.
    Return expanded set. Limits by max_recursive nodes.
    """
    resolved_all: Set[Path] = set(initial_headers)
    q = deque(initial_headers)
    processed = 0
    if use_progress and HAS_TQDM:
        pbar = tqdm(total=0, desc="Recursing headers", unit="hdr")
    else:
        pbar = None

    while q and processed < max_recursive:
        h = q.popleft()
        processed += 1
        if pbar:
            pbar.total = processed + len(q) + 1  # non precise, but gives movement
            pbar.update(1)
        includes = parse_includes_from_header(h)
        for delim, inc in includes:
            # For <> style we search include dirs; for " " style we prefer relative to header
            candidate_paths = []
            if delim == '"':
                # relative to header dir first
                rel = h.parent / inc
                candidate_paths.append(rel)
            # always try resolver
            p = resolver.resolve(inc)
            if p:
                candidate_paths.append(p)
            # also try header dir / basename
            candidate_paths.append(h.parent / Path(inc).name)
            found = None
            for cp in candidate_paths:
                if isinstance(cp, Path) and cp.exists():
                    found = cp
                    break
            if found and found not in resolved_all:
                resolved_all.add(found)
                q.append(found)
        # safety: if too many processed, break
    if pbar:
        pbar.close()
    return resolved_all

def copy_files_preserve_structure(files: List[Path], src_root: Path, out_root: Path, use_progress: bool = True) -> Tuple[List[str], List[Tuple[str,str]]]:
    """
    Copy files preserving relative paths when under src_root.
    For files outside src_root, mirror their absolute path under __external__/
    """
    copied = []
    skipped = []
    it = files
    if use_progress and HAS_TQDM:
        it = tqdm(sorted(files), desc="Copying files", unit="file")
    for f in it:
        try:
            # Try to make relative path under src_root
            rel = safe_rel(f, src_root)
            dest = out_root / rel
        except ValueError:
            # File outside src_root → put under __external__/abs_path
            rel = Path("__external__") / safe_rel(f, Path("/"))
            dest = out_root / rel

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(f), str(dest))
            copied.append(str(rel))
        except Exception as e:
            skipped.append((str(f), str(e)))
    return copied, skipped

# -----------------------
# CLI and main
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract used kernel source and headers (from .cmd).")
    p.add_argument("--src", required=True, help="Path to kernel source root")
    p.add_argument("--out", required=True, help="Output directory to copy used files into")
    p.add_argument("--build-dir", help="If build output (O=...) dir is separate, point here to find .cmd files")
    p.add_argument("--recursive-headers", action="store_true",
                   help="Recursively resolve headers' included headers (can expand a lot)")
    p.add_argument("--top-n", type=int, default=60, help="Top N nodes for graph visualization")
    p.add_argument("--max-search", type=int, default=500, help="Limit for basename search")
    p.add_argument("--max-recursive", type=int, default=10000, help="Limit for recursive header traversal")
    p.add_argument("--no-graph", action="store_true", help="Don't attempt to build PNG graph even if networkx present")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def generate_dot(edges: List[Tuple[str,str]], out_dot: Path) -> None:
    out_dot.write_text("digraph deps {\n")
    with out_dot.open("a") as fo:
        for a, b in edges:
            fo.write(f'  "{a}" -> "{b}";\n')
        fo.write("}\n")

def generate_graph_png(edges: List[Tuple[str,str]], out_png: Path, top_n: int = 60) -> bool:
    if not HAS_NETWORKX:
        return False
    G = nx.DiGraph()
    G.add_edges_from(edges)
    deg = dict(G.degree())
    top_nodes = sorted(deg, key=lambda n: deg[n], reverse=True)[:top_n]
    H = G.subgraph(top_nodes).copy()
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(H, seed=42)
    nx.draw(H, pos, with_labels=True, node_size=300, font_size=6)
    plt.title("Top dependency graph (by degree)")
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close()
    return True

def safe_rel(path: Path, root: Path):
    """尝试计算相对路径，不在root下则放入 __external__ 区域"""
    try:
        return path.relative_to(root)
    except ValueError:
        return Path("__external__") / Path("/".join(path.parts[1:]))

def main():
    args = parse_args()
    src_root = Path(args.src).resolve()
    if args.build_dir:
        cmd_root = Path(args.build_dir).resolve()
    else:
        cmd_root = src_root
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists() or not src_root.is_dir():
        print(f"[ERROR] src path not found: {src_root}", file=sys.stderr)
        sys.exit(2)
    if not cmd_root.exists() or not cmd_root.is_dir():
        print(f"[ERROR] build/cmd path not found: {cmd_root}", file=sys.stderr)
        sys.exit(2)

    print(f"[+] Kernel src: {src_root}")
    print(f"[+] Searching .cmd under: {cmd_root}")
    print(f"[+] Output dir: {out_root}")
    print(f"[+] Options: recursive_headers={args.recursive_headers}, top_n={args.top_n}, max_search={args.max_search}")

    # find cmd files
    cmd_files = find_cmd_files(cmd_root)
    print(f"[+] Found {len(cmd_files)} .cmd files")

    resolver = Resolver(src_root, max_search=args.max_search, verbose=args.verbose)

    tokens = gather_tokens(cmd_files, use_progress=True)
    print(f"[+] Extracted {len(tokens)} tokens from .cmd files")

    srcs, hdrs, unresolved = resolve_tokens(tokens, resolver, use_progress=True)
    print(f"[+] Resolved sources: {len(srcs)}, headers: {len(hdrs)}, unresolved tokens: {len(unresolved)}")

    # build mapping source -> headers
    src_to_hdrs = build_src_to_headers_map(cmd_files, resolver, use_progress=True)

    # if some srcs not in mapping, ensure they exist
    for s in srcs:
        src_to_hdrs.setdefault(s, set())

    # Optionally do recursive header inclusion
    if args.recursive_headers:
        print("[*] Expanding headers recursively (this may be large)...")
        initial_headers = set(hdrs)
        expanded = recursive_header_resolution(initial_headers, resolver, max_recursive=args.max_recursive, use_progress=True)
        new_headers = sorted(expanded)
        print(f"[+] After recursion headers count: {len(new_headers)} (was {len(hdrs)})")
        hdrs = new_headers
        # add newly discovered headers into mapping (no per-source info for recursive ones)
        # but we will keep src_to_hdrs as is (direct headers), and use hdrs as extra files to copy

    # assemble list of files to copy
    all_used_files = set(srcs) | set(hdrs)
    # also include headers referenced by per-source mapping
    for s, hs in src_to_hdrs.items():
        all_used_files.update(hs)

    print(f"[+] Total unique files to copy: {len(all_used_files)}")

    copied, skipped = copy_files_preserve_structure(list(all_used_files), src_root, out_root, use_progress=True)
    print(f"[+] Copied files: {len(copied)}, skipped: {len(skipped)}")

    # save extracted_files list
    extracted_list_path = out_root / "extracted_files.txt"
    with extracted_list_path.open("w") as fo:
        for p in sorted(all_used_files):
            try:
                rel = safe_rel(p, src_root)
            except ValueError:
                # 文件在源码树外部 → 写入带 __external__ 前缀的路径
                rel = Path("__external__") / safe_rel(p, Path("/"))
            fo.write(str(rel) + "\n")

    # unresolved tokens
    unresolved_path = out_root / "unresolved_tokens.txt"
    with unresolved_path.open("w") as fo:
        for t in unresolved:
            fo.write(t + "\n")

    # mapping JSON (serialize paths relative to src)
    serial_map = {}
    for s, hs in src_to_hdrs.items():
        s_rel = safe_rel(s, src_root)
        h_rels = sorted(str(safe_rel(h, src_root)) for h in hs)
        serial_map[str(s_rel)] = h_rels
    deps_json = out_root / "deps.json"
    deps_json.write_text(json.dumps(serial_map, indent=2))

    # dot file
    edges = []
    for s, hs in src_to_hdrs.items():
        srel = str(safe_rel(s, src_root))
        for h in hs:
            hrel = str(safe_rel(h, src_root))
            edges.append((srel, hrel))
    dot_path = out_root / "deps.dot"
    generate_dot(edges, dot_path)

    # try graph png
    png_path = out_root / "deps_top_graph.png"
    graph_created = False
    if not args.no_graph and HAS_NETWORKX and edges:
        try:
            graph_created = generate_graph_png(edges, png_path, top_n=args.top_n)
        except Exception as e:
            print(f"[!] Graph generation failed: {e}")

    # statistics
    top_level = Counter()
    top2 = Counter()
    for p in all_used_files:
        rel = safe_rel(p, src_root)
        parts = rel.parts
        if parts:
            top_level[parts[0]] += 1
        if len(parts) >= 2:
            top2[f"{parts[0]}/{parts[1]}"] += 1

    # write summary
    summary = {
        "kernel_src": str(src_root),
        "cmd_root": str(cmd_root),
        "out_root": str(out_root),
        "n_cmd_files": len(cmd_files),
        "n_tokens": len(tokens),
        "n_sources": len(srcs),
        "n_headers": len(hdrs),
        "n_unresolved_tokens": len(unresolved),
        "copied_files": len(copied),
        "skipped_files": len(skipped),
        "dot_file": str(dot_path),
        "png_file": str(png_path) if graph_created else None,
    }
    with (out_root / "summary.json").open("w") as fo:
        json.dump(summary, fo, indent=2)

    # print summary
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts in: {out_root}")
    print(" - extracted_files.txt")
    print(" - deps.json")
    print(" - deps.dot")
    print(" - unresolved_tokens.txt")
    print(" - summary.json")
    if graph_created:
        print(" - deps_top_graph.png (visualized top nodes)")

    # optionally display pretty table if pandas available
    if HAS_PANDAS:
        df_top = pd.DataFrame(top_level.items(), columns=["top_dir", "count"]).sort_values("count", ascending=False).reset_index(drop=True)
        print("\nTop-level counts:")
        print(df_top.head(40).to_string(index=False))

if __name__ == "__main__":
    main()

r'''
快速使用说明（示例）

把上面的内容保存为 extract_kernel_used_src.py，并赋可执行权限：

chmod +x extract_kernel_used_src.py


在你的内核源码树完成一次构建（必要，确保 .cmd 文件存在）：

若直接在源码树构建：

make -j$(nproc)


若用 O= 指定了输出目录（例：make O=build），要把 --build-dir 指向 build：

make O=build -j$(nproc)
python3 extract_kernel_used_src.py --src /path/to/linux --build-dir /path/to/linux/build --out ~/kernel_used_src


运行脚本（示例）：

python3 extract_kernel_used_src.py --src /home/user/linux-6.6 --out /home/user/kernel_used_src --recursive-headers --top-n 80


如果只想直接提取不递归头，去掉 --recursive-headers；

若你的 .cmd 文件在源码树（常见）， --build-dir 可省略。

输出会在 --out 指定目录下，包含 extracted_files.txt、deps.json、deps.dot、summary.json 等。

依赖建议（可选增强）

安装进度/可视化支持：

pip3 install tqdm networkx matplotlib pandas


tqdm：漂亮的进度条

networkx + matplotlib：生成 deps_top_graph.png

pandas：漂亮的统计输出（可选）
'''