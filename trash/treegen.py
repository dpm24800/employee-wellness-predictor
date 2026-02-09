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
IGNORE_FOLDERS: Set[str] = {"__pycache__", ".git", ".venv", "venv", ".idea", 
                            ".vscode", ".pytest_cache", "node_modules", "trash"
                            }


IGNORE_FILES: Set[str] = {".DS_Store", "Thumbs.db", ".gitignore", 
                            ".env", ".env.example"
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
def generate_tree(
    path: Path,
    prefix: str = "",
    is_last: bool = False,
    stats: dict = None
) -> List[str]:
    """
    Recursively generates tree lines with exact formatting:
    - Root folder name on its own line (no prefix)
    - Root children start at column 0 with ├──/└──
    - Subdirectories use proper indentation with │   and spaces
    """
    if stats is None:
        stats = {"folders": 0, "files": 0}
    
    lines: List[str] = []
    
    # Root node: just the folder name with trailing slash (no prefix)
    if prefix == "" and not is_last:
        folder_name = path.name if path.name else path.absolute().name
        lines.append(f"{folder_name}/")
        stats["folders"] += 1
    else:
        # Non-root nodes: use connector with prefix
        connector = "└── " if is_last else "├── "
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{prefix}{connector}{path.name}{suffix}")
        
        if path.is_dir():
            stats["folders"] += 1
        else:
            stats["files"] += 1
    
    # Process directory contents
    if path.is_dir():
        entries = get_sorted_entries(path)
        entry_count = len(entries)
        
        for idx, (entry_name, is_folder) in enumerate(entries):
            entry_path = path / entry_name
            is_last_entry = (idx == entry_count - 1)
            
            # Determine prefix for children
            if is_last:
                # Parent was last item → use spaces for children
                child_prefix = prefix + "    "
            else:
                # Parent had siblings below → use │ for children
                child_prefix = prefix + "│   "
            
            if is_folder:
                # Recurse into subdirectory
                lines.extend(generate_tree(entry_path, child_prefix, is_last_entry, stats))
            else:
                # Add file entry
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
    
    stats = {"folders": 0, "files": 0}
    tree_lines = generate_tree(root, stats=stats)
    
    # Build final output with exact format
    output = [
        f"Tree structure for: {root.name}/",
        "",
        *tree_lines,
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