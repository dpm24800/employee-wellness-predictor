#!/usr/bin/env python3
"""
Tree Structure Generator
Exact format: root folder name first with no indentation for its children
Sorting: special chars (_, .) → numbers → lowercase → uppercase
Folders always listed before files at each level
"""

import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

# ============================================
# CONFIGURATION
# ============================================

IGNORE_FOLDERS: Set[str] = {
    "__pycache__", ".git", ".venv", "venv", ".idea",
    ".vscode", ".pytest_cache", "node_modules"
}

IGNORE_FILES: Set[str] = {
    ".DS_Store", "Thumbs.db", ".gitignore",
    ".env", ".env.example", "treegen.py"
}

# ============================================
# SORTING LOGIC
# ============================================

def get_sort_key(name: str) -> Tuple[int, str]:
    """
    Sorting priority:
    0 = special char (_, ., -, etc.)
    1 = number (0-9)
    2 = lowercase letter (a-z)
    3 = uppercase letter (A-Z)
    """
    first_char = name[0]

    if first_char in ('_', '.', '-', '~', '#', '$', '@'):
        priority = 0
    elif first_char.isdigit():
        priority = 1
    elif first_char.islower():
        priority = 2
    elif first_char.isupper():
        priority = 3
    else:
        priority = 4

    return priority, name.lower()


def get_sorted_entries(path: Path) -> List[Tuple[str, bool]]:
    """
    Returns sorted entries as [(name, is_folder), ...]
    Folders always come before files.
    Respects ignore rules and skips symlinks/inaccessible entries.
    """

    try:
        entries = os.listdir(path)
    except (PermissionError, OSError) as e:
        print(f"⚠️ Warning: Cannot read {path}: {e}", file=sys.stderr)
        return []

    folders, files = [], []

    for entry in entries:
        if entry in IGNORE_FOLDERS or entry in IGNORE_FILES:
            continue

        full_path = path / entry

        try:
            if full_path.is_symlink():
                continue

            if full_path.is_dir():
                folders.append(entry)
            else:
                files.append(entry)

        except (OSError, PermissionError):
            continue

    folders.sort(key=get_sort_key)
    files.sort(key=get_sort_key)

    return [(f, True) for f in folders] + [(f, False) for f in files]


# ============================================
# TREE GENERATION
# ============================================

def generate_tree(
    path: Path,
    prefix: str = "",
    is_last: bool = True,
    stats: dict = None,
    is_root: bool = False
) -> List[str]:

    if stats is None:
        stats = {"folders": 0, "files": 0}

    lines: List[str] = []

    # ----- root node -----
    if is_root:
        folder_name = path.name if path.name else path.absolute().name
        lines.append(f"{folder_name}/")
        stats["folders"] += 1
    else:
        connector = "└── " if is_last else "├── "
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{prefix}{connector}{path.name}{suffix}")

        if path.is_dir():
            stats["folders"] += 1
        else:
            stats["files"] += 1

    # ----- children -----
    if path.is_dir():
        entries = get_sorted_entries(path)
        total = len(entries)

        for i, (entry_name, is_folder) in enumerate(entries):
            entry_path = path / entry_name
            last = (i == total - 1)

            # IMPORTANT:
            # root children must start with no │ prefix
            if is_root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("    " if is_last else "│   ")

            if is_folder:
                lines.extend(
                    generate_tree(
                        entry_path,
                        prefix=child_prefix,
                        is_last=last,
                        stats=stats,
                        is_root=False
                    )
                )
            else:
                connector = "└── " if last else "├── "
                lines.append(f"{child_prefix}{connector}{entry_name}")
                stats["files"] += 1

    return lines


# ============================================
# EXPORT
# ============================================

def export_tree(root_path: str = ".", output_file: str = "tree.txt") -> None:
    root = Path(root_path).resolve()

    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")

    stats = {"folders": 0, "files": 0}

    tree_lines = generate_tree(root, stats=stats, is_root=True)

    output = [
        f"Tree structure for: {root.name}/",
        "",
        *tree_lines,
        "",
        f"📊 Total: {stats['folders']} folders, {stats['files']} files"
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    print("\n".join(output))
    print(f"\n✅ Tree exported to `{output_file}`")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    export_tree(root_path=".", output_file="tree.txt")
