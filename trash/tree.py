import os

IGNORE_FOLDERS = {"__pycache__", ".git", ".venv", "venv"}

def generate_tree(path=".", prefix=""):
    tree_lines = []
    entries = sorted(os.listdir(path))
    entries = [e for e in entries if e not in IGNORE_FOLDERS]

    for index, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        tree_lines.append(prefix + connector + entry)

        if os.path.isdir(full_path):
            extension = "    " if is_last else "│   "
            tree_lines.extend(generate_tree(full_path, prefix + extension))

    return tree_lines


def export_tree(filename="tree.txt"):
    tree = generate_tree()
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(tree))

    print(f"✅ Folder tree exported to `{filename}`")


if __name__ == "__main__":
    export_tree("tree.md")   # change to "tree.txt" if needed