import sys, types

from src.dsl.dsl import *


def find_module_functions(module_name):
    """
    Yields (module, function_name, function) tuples for all functions in a module and its imports.

    First yields functions defined directly in the target module, then yields functions from
    other modules that may have been imported via star imports.

    Args:
        module_name: Name of the module to scan for functions

    Yields:
        Tuples of (module, function_name, function_object)
    """
    if module_name not in sys.modules:
        debug(sys.modules)
    module = sys.modules[module_name]
    # First yield functions from the target module
    for name, obj in list(module.__dict__.items()):
        if isinstance(obj, types.FunctionType) and obj.__module__ == module_name:
            yield (module, name, obj)

    # Then yield functions from other modules (for star imports)
    for other_module_name, other_module in list(sys.modules.items()):
        if other_module and other_module_name != module_name:
            for name, obj in list(getattr(other_module, "__dict__", {}).items()):
                # Only yield if it's a function that was defined in our target module
                if (
                    isinstance(obj, types.FunctionType)
                    and obj.__module__ == module_name
                ):
                    yield (other_module, name, obj)


if __name__ == "__main__":
    debug(list(find_module_functions("src.dsl.dsl")))  # ok
