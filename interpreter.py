"""
Interactive interpreter for ARC DSL with function tracing
"""

import code
import sys
import importlib
import json
import numpy as np
import typer
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import inspect

# Import local modules
from arc_types import *
from constants import *
from dsl import *
import main
from tracer import DSLTracer

# Task data directory
DATA_DIR = Path("data")

app = typer.Typer()


def list_dsl_functions() -> None:
    """List all available DSL functions"""
    dsl_funcs = [
        name
        for name, obj in inspect.getmembers(sys.modules["dsl"])
        if inspect.isfunction(obj) and not name.startswith("_")
    ]
    print("\nAvailable DSL functions:")
    print("------------------------")
    for func in sorted(dsl_funcs):
        print(f"- {func}")


def man_dsl_function(func_name: str) -> None:
    """Get type definition of a DSL function"""
    try:
        with open("out/dsl.pyi", "r") as f:
            pyi_contents = f.read()

        # Find the function definition using regex
        pattern = rf"def {func_name}\((.*?)\) -> (.*?):"
        match = re.search(pattern, pyi_contents, re.DOTALL)

        if match:
            args, return_type = match.groups()
            # Find docstring
            docstring_pattern = rf"def {func_name}.*?\n\s+\"\"\"(.*?)\"\"\""
            docstring_match = re.search(docstring_pattern, pyi_contents, re.DOTALL)
            docstring = (
                docstring_match.group(1).strip()
                if docstring_match
                else "No description available"
            )

            print(f"\nFunction: {func_name}")
            print("------------------------")
            print(f"Type: ({args}) -> {return_type}")
            print(f"Description: {docstring}")
        else:
            print(f"Function '{func_name}' not found in DSL")
    except FileNotFoundError:
        print("Type definitions file (dsl.pyi) not found")
    except Exception as e:
        print(f"Error getting function definition: {e}")


def load(task_id: str, idx: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a task's input grid and optionally its output

    Args:
        task_id: The task ID to load
        idx: Index of the example to load (0-based)

    Returns:
        Tuple of (input_grid, output_grid)
        Output grid may be None if idx is for a test example
    """
    # Use main.py's data loading
    data = main.get_data(train=True)

    if task_id not in data["train"]:
        raise ValueError(f"Task {task_id} not found")

    # Count total examples (train + test)
    train_examples = data["train"][task_id]
    test_examples = data["test"][task_id]
    all_examples = train_examples + test_examples

    # Check if index is valid
    if idx < 0 or idx >= len(all_examples):
        raise ValueError(f"Example index {idx} out of range (0-{len(all_examples)-1})")

    # Get the example
    example = all_examples[idx]

    # Convert to tuple of tuples
    input_grid = tuple(tuple(row) for row in example["input"])

    # Output may not exist for test examples
    output_grid = None
    if "output" in example:
        output_grid = tuple(tuple(row) for row in example["output"])

    return input_grid, output_grid


class DSLInterpreter(code.InteractiveConsole):
    """Interactive interpreter with DSL function tracing"""

    def __init__(self, task_id: Optional[str] = None):
        """Initialize the interpreter

        Args:
            task_id: Optional task ID to load initially
        """
        # Create a clean namespace
        local_ns = {}

        # Initialize the interpreter
        super().__init__(local_ns)

        # Initialize the tracer
        self.tracer = DSLTracer()
        self.current_task_id = None

        # Import DSL modules for the interpreter and instrument them
        for module_name in ["dsl", "constants", "arc_types"]:
            module = sys.modules[module_name]
            self.instrument_module(module)

        # Add helper functions to namespace
        self.locals["load"] = load
        self.locals["list"] = list_dsl_functions
        self.locals["man"] = man_dsl_function

        # Load task if specified
        if task_id:
            try:
                input_grid, _ = load(task_id)
                self.locals["I"] = input_grid
                self.current_task_id = task_id
                print(f"Loaded task {task_id}")
                print(f"Input grid 'I' shape: {input_grid.shape}")
            except Exception as e:
                print(f"Error loading task: {e}")

    def instrument_module(self, module):
        """Instrument a module and add it to the interpreter namespace"""
        # First instrument it
        self.tracer.instrument_module(module.__name__)

        # Then add the instrumented versions to our namespace
        for name, obj in module.__dict__.items():
            if not name.startswith("_"):  # Skip private members
                self.locals[name] = obj

    def runsource(self, source, filename="<input>", symbol="single"):
        """Override to add tracing for all executed code"""
        # Detect solver function definitions to auto-instrument
        solver_match = re.match(r"def\s+solve_([a-zA-Z0-9_]+)\s*\(", source)
        if solver_match:
            # Auto-load the task if it exists
            task_id = solver_match.group(1)
            if not self.current_task_id or self.current_task_id != task_id:
                try:
                    input_grid, _ = load(task_id)
                    self.locals["I"] = input_grid
                    self.current_task_id = task_id
                    print(f"Loaded task {task_id}")
                    print(f"Input grid 'I' shape: {input_grid.shape}")
                except Exception:
                    # If task not found, continue anyway
                    pass

        # Clear trace for each new execution
        self.tracer.clear()

        # Enable tracing
        self.tracer.enable()

        try:
            # Try to execute the code
            code_obj = code.compile_command(source, filename, symbol)
            if code_obj is None:
                # Incomplete input
                return True

            # Execute the code
            self.runcode(code_obj)

            # Process the trace
            self._process_trace()

            return False
        except (OverflowError, SyntaxError, ValueError):
            # Show syntax errors
            self.showsyntaxerror(filename)
            return False
        except Exception:
            # Show other errors
            self.showtraceback()
            return False

    def _process_trace(self):
        """Process and display the trace information"""
        # Get function stats
        stats = self.tracer.get_function_stats()

        if stats:
            print("\n--- Function Call Statistics ---")
            for name, data in sorted(
                stats.items(), key=lambda x: x[1]["count"], reverse=True
            ):
                print(f"{name}: {data['count']} calls, {data['avg_time_ms']:.2f}ms avg")
                if data["grid_changes"]:
                    print(f"  Grid changes: {data['grid_changes']}")

            # Print function calls with grid changes
            grid_change_calls = [
                call
                for call in self.tracer.execution_log
                if call.grid_changes and call.grid_changes > 0
            ]

            if grid_change_calls:
                print("\n--- Grid Transformations ---")
                for i, call in enumerate(grid_change_calls[:5]):
                    print(f"{call.function_name}: {call.grid_changes} changes")
                    # Print first 5 coordinate changes
                    for j, (i, j, old_val, new_val) in enumerate(call.diff_coords[:5]):
                        print(f"  ({i},{j}): {old_val} -> {new_val}")

                    if len(call.diff_coords) > 5:
                        print(f"  ... and {len(call.diff_coords) - 5} more changes")

            # Show total call count
            print(f"\nTotal function calls: {len(self.tracer.execution_log)}")


@app.command()
def run(task_id: Optional[str] = None):
    """Run the DSL interpreter with tracing"""
    interpreter = DSLInterpreter(task_id)

    # Create welcome message
    print("DSL Interpreter with Tracing")
    print("---------------------------")
    print("Type Python/DSL code to evaluate")
    print("Available commands:")
    print("  load(task_id, idx=0): Load a specific task and example")
    print("  list(): List all available DSL functions")
    print("  man(func_name): Get type definition and description of a DSL function")
    print("  exit(): Exit the interpreter")

    # Start the interpreter
    interpreter.interact(banner="")


if __name__ == "__main__":
    app()
