"""Tool config page."""

import importlib.util
import sys
from types import ModuleType
import streamlit as st
from code_editor import code_editor
from pydantic import BaseModel
from aic_core.agent.agent_hub import AgentHub
from aic_core.streamlit.mixins import ToolSelectorMixin
from aic_core.streamlit.page import AICPage


class ToolConfigPage(AICPage, ToolSelectorMixin):
    """Tool config page."""

    def __init__(self, repo_id: str) -> None:
        """Initialize the page."""
        super().__init__()
        self.repo_id = repo_id

    def load_code_as_module(
        self, code_text: str, module_name: str = "dynamic_module"
    ) -> ModuleType:
        """Load code text as a Python module.

        Args:
            code_text (str): The Python code as a string
            module_name (str): Name to give the module

        Returns:
            module: The loaded Python module
        """
        # Create a module spec
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        assert spec

        # Create a new module based on the spec
        module = importlib.util.module_from_spec(spec)

        # Add the module to sys.modules
        sys.modules[module_name] = module

        # Execute the code within the module's namespace
        exec(code_text, module.__dict__)

        return module

    def edit_function(self, function_name: str) -> None:
        """Edit function."""
        hf_repo = AgentHub(self.repo_id)
        if function_name:
            file_path = hf_repo.get_file_path(function_name, AgentHub.tools_dir)
            with open(file_path) as f:
                default_code = f.read()
        else:
            with open("extendable_agents/app/extension_template.py") as f:
                default_code = f.read()

        code = code_editor(
            default_code, lang="python", height=300, options={"wrap": True}
        )

        if not code["text"]:
            st.warning(
                "Not saved. "
                "Press `Control + Enter` (Windows) or `Command + Enter` (Mac) to save."
            )
        function_name = st.text_input(
            "Function or Pydantic model name",
            value=function_name,
            disabled=not code["text"],
        )

        button_disabled = not function_name or not code["text"]
        if st.button("Save", disabled=button_disabled):
            # Test loading the code as a module
            if code["text"]:
                try:
                    dynamic_module = self.load_code_as_module(code["text"])
                    module_contents = [
                        item
                        for item in dir(dynamic_module)
                        if not item.startswith("__")
                    ]
                    assert function_name in module_contents
                    func = getattr(dynamic_module, function_name)
                    # Upload the code to Tools Hub
                    if isinstance(func, type) and issubclass(func, BaseModel):
                        hf_repo.upload_content(
                            function_name, code["text"], AgentHub.pydantic_models_dir
                        )
                    else:
                        hf_repo.upload_content(
                            function_name, code["text"], AgentHub.tools_dir
                        )
                    st.success(f"Function `{function_name}` saved successfully!")
                except AssertionError:
                    st.error(f"Definition `{function_name}` not found in module")
                except Exception as e:
                    st.error(f"Error loading code as module: {str(e)}")
            else:
                st.write()

    def run(self) -> None:
        """Main function."""
        st.title("Custom Function or Pydantic Model")
        tools = self.tool_selector(self.repo_id)
        selected_tool = tools[0] if tools else ""
        self.edit_function(selected_tool)
