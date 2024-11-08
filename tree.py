import os

# List of top-level folders to create in the current directory
folders = [
    "config",
    "data",
    "docs",
    "models",
    "notebooks",
    "utils",
]

# Step 1: Create each folder in the list in the current directory
for folder in folders:
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist

# Step 2: Generate a tree structure and write it to docs/project_structure.txt
def generate_tree(root_dir, indent=""):
    tree_str = ""
    items = sorted(os.listdir(root_dir))

    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = i == len(items) - 1
        tree_str += f"{indent}{'└── ' if is_last else '├── '}{item}\n"
    return tree_str

# Ensure the docs folder exists before writing the tree structure
os.makedirs("docs", exist_ok=True)

# Write the folder structure to docs/project_structure.txt
with open("docs/project_structure.txt", "w") as file:
    file.write("Project Structure:\n")
    file.write(generate_tree("."))
