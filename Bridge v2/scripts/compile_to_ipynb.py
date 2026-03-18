import ast
import json
import os
from pathlib import Path


def get_imports_and_code(file_path, local_modules, is_main_file=False):
    """Extract imports and code from a Python file, removing local module imports and replacing module.function() calls."""

    with open(file_path, "r") as f:
        content = f.read()

    tree = ast.parse(content)
    code_lines = content.split("\n")
    lines_to_skip = set()

    if is_main_file:
        for i, line in enumerate(code_lines):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                break

            lines_to_skip.add(i)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module in local_modules:

                for line_num in range(node.lineno - 1, node.end_lineno):
                    lines_to_skip.add(line_num)

            elif node.module == "main":
                for line_num in range(node.lineno - 1, node.end_lineno):
                    lines_to_skip.add(line_num)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in local_modules:

                    for line_num in range(node.lineno - 1, node.end_lineno):
                        lines_to_skip.add(line_num)

    filtered_lines = []
    skip_main = False

    for i, line in enumerate(code_lines):
        if i in lines_to_skip:
            continue

        if "if __name__" in line and "__main__" in line:
            skip_main = True
        elif skip_main and line and not line[0].isspace():
            skip_main = False

        if not skip_main:
            modified_line = line
            for module in local_modules:
                import re as voodoo

                pattern = r"\b" + voodoo.escape(module) + r"\.([a-zA-Z_][a-zA-Z0-9_]*)"
                if module in modified_line and module + "." in modified_line:
                    parts = modified_line.split(module + ".")

                    if len(parts) > 1:
                        before = parts[0]

                        if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                            modified_line = voodoo.sub(pattern, r"\1", modified_line)

            filtered_lines.append(modified_line)

    return "\n".join(filtered_lines)


def find_local_imports(file_path, project_root):
    """Find all local module imports."""

    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    local_imports = []

    for node in ast.walk(tree):

        if isinstance(node, ast.ImportFrom):
            if node.module and not node.level:
                module_path = Path(project_root) / node.module.replace(".", "/") / "__init__.py"

                if not module_path.exists():
                    module_path = Path(project_root) / f"{node.module.replace('.', '/')}.py"

                if module_path.exists():
                    local_imports.append(str(module_path))

        elif isinstance(node, ast.Import):
            for alias in node.names:

                module_name = alias.name

                module_path = Path(project_root) / module_name.replace(".", "/") / "__init__.py"

                if not module_path.exists():
                    module_path = Path(project_root) / f"{module_name.replace('.', '/')}.py"

                if module_path.exists():
                    local_imports.append(str(module_path))

    return local_imports


def create_notebook(main_file, output_file="notebook.ipynb"):
    """Create Jupyter notebook from main file and its dependencies."""

    project_root = Path(main_file).parent

    visited = set()

    to_process = [main_file]

    dependency_graph = {}

    while to_process:
        current = to_process.pop(0)

        if current in visited:
            continue

        visited.add(current)
        deps = find_local_imports(current, project_root)
        dependency_graph[current] = deps
        to_process.extend([d for d in deps if d not in visited])

    sorted_files = []

    temp_visited = set()

    perm_visited = set()

    def visit(node):

        if node in perm_visited:
            return

        if node in temp_visited:
            return

        temp_visited.add(node)

        for dep in dependency_graph.get(node, []):
            visit(dep)

        temp_visited.remove(node)
        perm_visited.add(node)
        sorted_files.append(node)

    for file in dependency_graph:
        visit(file)

    local_modules = set()

    for file_path in sorted_files:
        module_name = Path(file_path).stem
        local_modules.add(module_name)

    constants_code = []

    with open(main_file, "r") as f:
        for line in f:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                break
            constants_code.append(line.rstrip())

    cells = []

    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Install dependencies\n",
                "!pip install -q torch torchvision torchaudio numpy pandas matplotlib pillow",
            ],
            "execution_count": None,
            "outputs": [],
        }
    )

    if constants_code:

        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Global Constants\n",
                    "Constants defined in main.py that are used by multiple modules.",
                ],
            }
        )

        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": [line + "\n" for line in constants_code if line.strip()],
                "execution_count": None,
                "outputs": [],
            }
        )

    for file_path in sorted_files:

        is_main = file_path == main_file
        code = get_imports_and_code(file_path, local_modules, is_main_file=is_main)

        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": [f"## {Path(file_path).name}"]}
        )

        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": code.split("\n"),
                "execution_count": None,
                "outputs": [],
            }
        )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(output_file, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Notebook created: {output_file}")


if __name__ == "__main__":
    main_file = Path(__file__).parent.parent / "main.py"
    output_file = Path(__file__).parent.parent / "colab_notebook.ipynb"
    create_notebook(str(main_file), str(output_file))
