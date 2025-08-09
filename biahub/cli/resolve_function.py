import numpy as np
import ultrack

# List of modules to scan for functions
VALID_MODULES = {"np": np, "ultrack.imgproc": ultrack.imgproc}

# Dynamically populate FUNCTION_MAP with functions from VALID_MODULES
FUNCTION_MAP = {
    f"{module_name}.{func}": getattr(module, func)
    for module_name, module in VALID_MODULES.items()
    for func in dir(module)
    if callable(getattr(module, func))
    and not func.startswith("__")  # Only include functions, not attributes
}


def resolve_function(function_name: str, custom_functions: dict = None):
    """
    Resolve a function by its name from the predefined FUNCTION_MAP.

    This function looks up a string identifier in a centralized dictionary of allowed
    functions and returns the corresponding callable. It is used to dynamically map
    function names (e.g., from config files) to actual Python functions.

    Parameters
    ----------
    function_name : str
        The fully qualified name of the function to retrieve
        (e.g., "np.mean", "ultrack.imgproc.gradient_magnitude").

    Returns
    -------
    Callable
        The resolved function object.

    Raises
    ------
    ValueError
        If the function name is not found in the `FUNCTION_MAP`.

    Notes
    -----
    - `FUNCTION_MAP` is a global dictionary that includes a whitelist of safe,
      user-approved or library-provided functions.
    - Additional functions (e.g., custom preprocessing functions) can be manually added
      to `FUNCTION_MAP`.
    """
    if custom_functions is not None:
        FUNCTION_MAP.update(custom_functions)

    if function_name not in FUNCTION_MAP:
        raise ValueError(
            f"Invalid function '{function_name}'. Allowed functions: {list(FUNCTION_MAP.keys())}"
        )

    return FUNCTION_MAP[function_name]
