import ast
import json
from pathlib import Path
import typer
from src.utils.devtools import debug

out_dir = Path(__file__).parent / "out"
writepath = out_dir / "dsl_typedefs.json"


def main() -> dict:
    """Extract DSL typedefs from .pyi files"""
    # Extract functions and types from all .pyi files
    all_functions = set()
    all_types = set()

    for fp in out_dir.glob("*.pyi"):
        with fp.open() as f:
            tree = ast.parse(f.read())

        all_functions.update(
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        )

        # Get type names from assignments like "TypeName = ..."
        all_types.update(
            target.id  # Name node's identifier
            for node in tree.body  # Top-level nodes
            if isinstance(node, ast.Assign)  # Assignment statements
            for target in node.targets  # LHS of assignment
            if isinstance(target, ast.Name)  # Simple name, not attribute etc
        )

    data = {"functions": list(all_functions), "types": list(all_types)}

    # Cache results
    with writepath.open("w") as f:
        json.dump(data, f, indent=2)

    return data


app = typer.Typer()


@app.command()
def list_functions():
    debug(main())


if __name__ == "__main__":
    app()
