import os
import importlib
import toml

_EXPECTED_FUNCTIONS = [
    "train_model",
    "test_model",
    "visualize_test",
    "test_model_integrity",
]

# Read configuration from pyproject.toml
def get_current_task():
    try:
        pyproject_data = toml.load("pyproject.toml")
        autobm_config = pyproject_data.get("tool", {}).get("autobm", {})
        return autobm_config.get("current_task", "ulti")
    except Exception as e:
        print(f"Warning: Could not read current_task from pyproject.toml: {e}")
        return "ulti"  # Default fallback

PIPELINE_TASK_NAME = get_current_task()

if not PIPELINE_TASK_NAME:
    raise ImportError(
        "The 'current_task' configuration in pyproject.toml is not set. "
        "Please set it to one of: 'ulti', 'rps', 'cda'"
    )

try:
    # 1. Construct the module name RELATIVE to the current package.
    #    The leading dot '.' is crucial for a relative import.
    relative_module_name = f".implementations.{PIPELINE_TASK_NAME}_pipeline"
    
    # 2. Use importlib.import_module with the 'package' argument.
    #    __package__ refers to the package the current module (pipeline.py) belongs to.
    #    If pipeline.py is in src/autobm/model/, then __package__ should be 'src.autobm.model'.
    print(f"Attempting to import '{relative_module_name}' from package '{__package__}'") # Debug print
    task_pipeline_module = importlib.import_module(relative_module_name, package=__package__)

    for func_name in _EXPECTED_FUNCTIONS:
        if hasattr(task_pipeline_module, func_name):
            globals()[func_name] = getattr(task_pipeline_module, func_name)
        else:
            raise AttributeError(
                f"Function '{func_name}' not found in module '{task_pipeline_module.__name__}'. " # Use the actual module name
                f"Ensure all expected pipeline functions ({_EXPECTED_FUNCTIONS}) "
                f"are defined for task '{PIPELINE_TASK_NAME}'."
            )
            
except ImportError as e:
    resolved_package_context = __package__ if __package__ else "None (is this script run as part of a package?)"
    raise ImportError(
        f"Could not import pipeline module for task '{PIPELINE_TASK_NAME}' using relative path '{relative_module_name}' from package context '{resolved_package_context}'. "
        f"Original error: {e}. \nMake sure the implementation file "
        f"(e.g., src/autobm/model/implementations/{PIPELINE_TASK_NAME}_pipeline.py) exists and "
        f"the 'CURRENT_TASK' environment variable ('{PIPELINE_TASK_NAME}') is correct."
    ) from e 
except AttributeError: # This will catch the AttributeError raised above
    raise

__all__ = _EXPECTED_FUNCTIONS