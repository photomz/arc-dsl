"""
DSL tracer for ARC-DSL
Provides runtime tracing of DSL functions without modifying their code
"""

from dataclasses import dataclass, field
import time
import sys
import types
import functools
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Tuple


@dataclass
class DSLFunctionCall:
    """Information about a single DSL function call"""

    id: int
    function_name: str
    args: tuple
    kwargs: dict
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    status: str = "pending"  # pending, success, error
    parent_id: Optional[int] = None
    line_number: Optional[int] = None
    file_name: Optional[str] = None
    # Grid-specific metrics
    input_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, int]] = None
    grid_changes: Optional[int] = None
    diff_coords: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )  # [(i, j, old_val, new_val), ...]

    def __post_init__(self):
        """Calculate duration once end_time is set"""
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def add_diff_info(self, args, result):
        """Add grid difference information if applicable"""
        # Check if result is grid-like (np.ndarray or tuple of tuples)
        if not isinstance(result, np.ndarray) and not (
            isinstance(result, tuple) and all(isinstance(row, tuple) for row in result)
        ):
            return

        # Convert to numpy array if it's a tuple of tuples
        if isinstance(result, tuple):
            result_array = np.array(result)
        else:
            result_array = result

        self.output_shape = result_array.shape

        # Look for grid-like inputs
        for arg in args:
            # Handle tuple of tuples format (arc-dsl native format)
            if isinstance(arg, tuple) and all(isinstance(row, tuple) for row in arg):
                arg_array = np.array(arg)
            elif isinstance(arg, np.ndarray):
                arg_array = arg
            else:
                continue

            if arg_array.ndim == 2:
                self.input_shape = arg_array.shape

                # If shapes match, calculate differences
                if arg_array.shape == result_array.shape:
                    # Find positions where grids differ
                    diff_mask = arg_array != result_array
                    diff_coords = list(zip(*diff_mask.nonzero()))
                    self.grid_changes = len(diff_coords)

                    # Store actual changes (limited to first 100)
                    for i, j in diff_coords[:100]:  # Limit to prevent huge logs
                        self.diff_coords.append(
                            (i, j, int(arg_array[i, j]), int(result_array[i, j]))
                        )
                break


@dataclass
class DSLTracer:
    """Traces all function calls in a DSL module"""

    execution_log: List[DSLFunctionCall] = field(default_factory=list)
    call_stack: List[int] = field(default_factory=list)
    call_counter: int = 0
    enabled: bool = True

    def enable(self):
        """Enable tracing"""
        self.enabled = True

    def disable(self):
        """Disable tracing"""
        self.enabled = False

    def clear(self):
        """Clear execution log"""
        self.execution_log = []
        self.call_counter = 0

    def wrap_function(self, func):
        """Wrap a function to trace its execution"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            # Get caller information
            frame = inspect.currentframe().f_back

            # Create function call record
            call_id = self.call_counter
            self.call_counter += 1

            parent_id = self.call_stack[-1] if self.call_stack else None

            call_info = DSLFunctionCall(
                id=call_id,
                function_name=func.__name__,
                args=args,
                kwargs=kwargs,
                start_time=time.time(),
                parent_id=parent_id,
                line_number=frame.f_lineno,
                file_name=frame.f_code.co_filename,
            )

            # Push to call stack
            self.call_stack.append(call_id)

            # Execute function
            try:
                result = func(*args, **kwargs)
                call_info.status = "success"
                call_info.result = result

                # Add grid diff information if applicable
                call_info.add_diff_info(args, result)

                return result
            except Exception as e:
                call_info.status = "error"
                call_info.error = str(e)
                raise
            finally:
                # Update timing information
                call_info.end_time = time.time()

                # Record function call
                self.execution_log.append(call_info)

                # Pop from call stack
                self.call_stack.pop()

        return wrapper

    def instrument_module(self, module_name):
        """Instrument all functions in a module"""
        debug(f"Instrumenting module: {module_name}")
        module = sys.modules[module_name]

        # Track instrumented functions to handle star imports
        instrumented_funcs = {}

        for name, obj in list(module.__dict__.items()):
            # Only wrap functions defined in this module
            if isinstance(obj, types.FunctionType) and obj.__module__ == module_name:
                wrapped = self.wrap_function(obj)
                module.__dict__[name] = wrapped
                instrumented_funcs[obj] = wrapped

        # Handle star imports by checking all modules
        for other_module_name, other_module in list(sys.modules.items()):
            if other_module and other_module_name != module_name:
                for name, obj in list(getattr(other_module, "__dict__", {}).items()):
                    # If this is one of our original functions that was star-imported
                    if (
                        isinstance(obj, types.FunctionType)
                        and obj in instrumented_funcs
                    ):
                        other_module.__dict__[name] = instrumented_funcs[obj]

        instrumented = list(instrumented_funcs.keys())
        print(f"Instrumenting: {instrumented[:4]} ... (total: {len(instrumented)})")

    def get_function_stats(self):
        """Return statistics about function calls"""
        stats = {}

        debug(self.execution_log)

        for call in self.execution_log:
            name = call.function_name
            if name not in stats:
                stats[name] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "avg_time_ms": 0,
                    "error_count": 0,
                    "grid_changes": 0,
                }

            stats[name]["count"] += 1
            if call.duration_ms:
                stats[name]["total_time_ms"] += call.duration_ms

            if call.status == "error":
                stats[name]["error_count"] += 1

            if call.grid_changes:
                stats[name]["grid_changes"] += call.grid_changes

        # Calculate averages
        for name, data in stats.items():
            if data["count"] > 0:
                data["avg_time_ms"] = data["total_time_ms"] / data["count"]

        return stats

    def get_call_tree(self):
        """Return a nested call tree"""
        # Build a map of parent -> children
        children_map = {}
        for call in self.execution_log:
            parent_id = call.parent_id
            if parent_id is not None:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(call.id)

        # Find root calls
        root_calls = [call for call in self.execution_log if call.parent_id is None]

        # Build tree recursively
        def build_tree(call):
            call_id = call.id
            children_ids = children_map.get(call_id, [])
            children = [
                build_tree(self.execution_log[child_id]) for child_id in children_ids
            ]

            return {
                "id": call.id,
                "function": call.function_name,
                "duration_ms": call.duration_ms,
                "status": call.status,
                "children": children,
            }

        return [build_tree(call) for call in root_calls]
