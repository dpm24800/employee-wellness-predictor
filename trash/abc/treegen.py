#!/usr/bin/env python3
"""
Tree Structure Generator
Exact format matching your specification:
- Root folder name on its own line (no prefix)
- Root children start at column 0 with ├──/└── (no leading whitespace)
- Sorting: special chars (_, .) → numbers → lowercase → uppercase
- Folders always listed before files at each level
"""
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

# ============================================
# CONFIGURATION
# ============================================
IGNORE_FOLDERS: Set[str] = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode", ".pytest_cache", 
                            "node_modules"
                        }

IGNORE_FILES: Set[str] = {".DS_Store", "Thumbs.db", ".gitignore", ".env", ".env.example", ".gitkeep"
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
    return (priority, name.lower())

def get_sorted_entries(path: Path) -> List[Tuple[str, bool]]:
    """
    Returns sorted entries as [(name, is_folder), ...] with folders before files.
    Respects ignore patterns and skips symlinks/inaccessible entries.
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
    
    # Sort with custom priority
    folders.sort(key=get_sort_key)
    files.sort(key=get_sort_key)
    
    return [(f, True) for f in folders] + [(f, False) for f in files]

# ============================================
# TREE GENERATION
# ============================================
def generate_tree_lines(path: Path, prefix: str, is_last: bool, stats: dict) -> List[str]:
    """
    Recursively generates tree lines for a directory entry.
    - prefix: indentation prefix for this level (e.g., "│   " or "    ")
    - is_last: whether this entry is the last child of its parent
    """
    lines = []
    
    # Add current entry line
    connector = "└── " if is_last else "├── "
    suffix = "/" if path.is_dir() else ""
    lines.append(f"{prefix}{connector}{path.name}{suffix}")
    
    # Update stats
    if path.is_dir():
        stats["folders"] += 1
    else:
        stats["files"] += 1
    
    # Process children if directory
    if path.is_dir():
        entries = get_sorted_entries(path)
        n = len(entries)
        
        # Determine prefix extension for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        for i, (entry_name, is_dir) in enumerate(entries):
            entry_path = path / entry_name
            is_last_entry = (i == n - 1)
            
            if is_dir:
                lines.extend(generate_tree_lines(entry_path, child_prefix, is_last_entry, stats))
            else:
                connector = "└── " if is_last_entry else "├── "
                lines.append(f"{child_prefix}{connector}{entry_name}")
                stats["files"] += 1
    
    return lines

def export_tree(root_path: str = ".", output_file: str = "tree.txt") -> None:
    """
    Generates and exports tree structure with exact formatting requirements.
    """
    root = Path(root_path).resolve()
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")
    
    stats = {"folders": 1, "files": 0}  # Root counts as 1 folder
    
    # Start with root folder name
    lines = [f"{root.name}/"]
    
    # Process root's children
    entries = get_sorted_entries(root)
    n = len(entries)
    
    for i, (entry_name, is_dir) in enumerate(entries):
        entry_path = root / entry_name
        is_last = (i == n - 1)
        
        if is_dir:
            lines.extend(generate_tree_lines(entry_path, "", is_last, stats))
        else:
            connector = "└── " if is_last else "├── "
            lines.append(f"{connector}{entry_name}")
            stats["files"] += 1
    
    # Build final output
    output = [
        f"Tree structure for: {root.name}/",
        "",
        *lines,
        "",
        f"📊 Total: {stats['folders']} folders, {stats['files']} files"
    ]
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    
    # Print to console
    print("\n".join(output))
    print(f"\n✅ Tree exported to `{output_file}`")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    export_tree(root_path=".", output_file="tree.txt")