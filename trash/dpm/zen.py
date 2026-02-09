import os

FOLDER_COUNT = 0
FILE_COUNT = 0


def tree(dir_path, prefix=""):
    global FOLDER_COUNT, FILE_COUNT

    entries = sorted(os.listdir(dir_path))
    entries = [e for e in entries if not e.startswith(".")]

    for index, entry in enumerate(entries):
        full_path = os.path.join(dir_path, entry)
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "

        print(prefix + connector + entry + ("/" if os.path.isdir(full_path) else ""))

        if os.path.isdir(full_path):
            FOLDER_COUNT += 1
            extension = "    " if is_last else "│   "
            tree(full_path, prefix + extension)
        else:
            FILE_COUNT += 1


if __name__ == "__main__":
    root = os.path.basename(os.getcwd())

    # print only the root (no extra first line)
    print(root + "/")

    tree(os.getcwd())

    print(f"\n📊 Total: {FOLDER_COUNT} folders, {FILE_COUNT} files")
