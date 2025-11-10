import importlib.resources
import os
import toml
from jinja2 import Environment, FileSystemLoader, meta

class PromptManager:
    def __init__(self, templates_subdir="templates", specific_root_dir="specific"):
        if __package__ is None:
            raise ImportError(
                "PromptManager should be imported as part of a package to reliably locate resources."
            )

        # Read current_task from pyproject.toml
        try:
            pyproject_data = toml.load("pyproject.toml")
            autobm_config = pyproject_data.get("tool", {}).get("autobm", {})
            current_task = autobm_config.get("current_task", "ulti")
        except Exception as e:
            print(f"Warning: Could not read current_task from pyproject.toml: {e}")
            current_task = "ulti"  # Default fallback

        if not current_task:
            raise EnvironmentError(
                "Configuration 'current_task' in pyproject.toml is not set or is empty. "
                "This is required to determine the specific files subdirectory."
            )

        self.effective_specific_subdir = f"{specific_root_dir}/{current_task}"

        try:
            templates_path_obj = importlib.resources.files(__package__) / templates_subdir
            self.env = Environment(loader=FileSystemLoader(str(templates_path_obj)))
            
            self._specific_files_root = importlib.resources.files(__package__) / self.effective_specific_subdir
        except Exception as e:
            raise FileNotFoundError(
                f"Could not initialize PromptManager: Ensure '{templates_subdir}' (for templates) and "
                f"'{self.effective_specific_subdir}' (for specific files under task '{current_task}') "
                f"directories exist within package '{__package__}'. Detailed error: {e}"
            )

    def _read_specific_file(self, filename: str) -> str | None:
        try:
            file_path_obj = self._specific_files_root / filename
            if file_path_obj.is_file():
                return file_path_obj.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return None 

    def render_prompt(self, template_name: str, **kwargs) -> str:
        try:
            template_source = self.env.loader.get_source(self.env, template_name)[0]
            parsed_content = self.env.parse(template_source)
            all_vars_in_template = meta.find_undeclared_variables(parsed_content)
        except Exception as e:
            raise ValueError(f"Failed to load or parse template '{template_name}': {e}") from e

        for var_name in all_vars_in_template:
            if var_name not in kwargs:
                file_content = self._read_specific_file(f"{var_name}.txt")
                if file_content is not None:
                    kwargs[var_name] = file_content

        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to render template '{template_name}': {e}") from e