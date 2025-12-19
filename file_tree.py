#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import fnmatch

BRANCH = "├── "
LASTBR = "└── "
VERT   = "│   "
SPACE  = "    "

DEFAULT_IGNORES = [".git", "__pycache__", ".DS_Store"]

def normalized_children(path, ignores, follow_symlinks):
    try:
        names = os.listdir(path)
    except PermissionError:
        return [], f"{path} [Permission denied]"
    except FileNotFoundError:
        return [], f"{path} [Not found]"
    # 过滤忽略项（支持通配符）
    if ignores:
        names = [
            n for n in names
            if not any(fnmatch.fnmatch(n, pat) for pat in ignores)
        ]
    # 按“目录优先、再按名称不区分大小写”排序
    names.sort(key=lambda n: (not os.path.isdir(os.path.join(path, n)) if not (follow_symlinks and os.path.islink(os.path.join(path, n))) else False,
                              n.lower()))
    return names, None

def draw_tree(root, max_depth=None, ignores=None, follow_symlinks=False):
    """
    返回用于写入文件的字符串列表（每行一项）
    """
    if ignores is None:
        ignores = DEFAULT_IGNORES

    lines = []
    root = os.path.abspath(root)
    root_display = os.path.basename(root.rstrip(os.sep)) or root
    lines.append(f"{root_display}/")

    def _recurse(cur_path, prefix, depth):
        if max_depth is not None and depth >= max_depth:
            return
        children, err = normalized_children(cur_path, ignores, follow_symlinks)
        if err:
            lines.append(prefix + BRANCH + err)
            return
        total = len(children)
        for idx, name in enumerate(children):
            full = os.path.join(cur_path, name)
            is_link = os.path.islink(full)
            try:
                is_dir = os.path.isdir(full) if (follow_symlinks or not is_link) else False
            except OSError:
                is_dir = False

            last = (idx == total - 1)
            branch = LASTBR if last else BRANCH
            suffix = "/"
            display = name + (suffix if is_dir else "")
            if is_link:
                try:
                    target = os.readlink(full)
                except OSError:
                    target = "?"
                display += f" -> {target}"

            lines.append(prefix + branch + display)

            if is_dir:
                _recurse(full, prefix + (SPACE if last else VERT), depth + 1)

    _recurse(root, "", 0)
    return lines

def main():
    parser = argparse.ArgumentParser(
        description="将根目录内容以 ASCII 树形结构导出到文件（类似 tree 命令的输出）。"
    )
    parser.add_argument("root", help="根目录路径")
    parser.add_argument("-o", "--output", default="tree.txt", help="输出文件路径（默认: tree.txt）")
    parser.add_argument("--max-depth", type=int, default=None, help="最大递归深度（默认不限制）")
    parser.add_argument("--ignore", default=",".join(DEFAULT_IGNORES),
                        help=f"以逗号分隔的忽略模式（支持通配符），默认: {','.join(DEFAULT_IGNORES)}")
    parser.add_argument("--follow-symlinks", action="store_true",
                        help="跟随目录符号链接（默认不跟随）")

    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.exists(root):
        print(f"路径不存在: {root}", file=sys.stderr)
        sys.exit(1)

    ignores = [p.strip() for p in args.ignore.split(",")] if args.ignore else []

    lines = draw_tree(
        root=root,
        max_depth=args.max_depth,
        ignores=ignores,
        follow_symlinks=args.follow_symlinks
    )

    # 同时在控制台打印，并写入文件（UTF-8 编码确保框线符号正常）
    text = "\n".join(lines) + "\n"
    print(text, end="")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n已保存到: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
