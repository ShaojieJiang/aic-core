"""Tool config page."""

import importlib.util
import os
import sys
from abc import abstractmethod
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

    def save_tool(self, tool_name: str, code: str) -> None:
        """Save tool."""
        hf_repo = AgentHub(self.repo_id)
        try:
            # Test loading the code as a module
            dynamic_module = self.load_code_as_module(code)
            module_contents = [
                item for item in dir(dynamic_module) if not item.startswith("__")
            ]
            assert tool_name in module_contents
            func = getattr(dynamic_module, tool_name)
            # Upload the code to Tools Hub
            if isinstance(func, type) and issubclass(func, BaseModel):
                hf_repo.upload_content(tool_name, code, AgentHub.result_types_dir)
            else:
                hf_repo.upload_content(tool_name, code, AgentHub.tools_dir)
        except AssertionError:
            st.error(f"Definition `{tool_name}` not found in module")
            st.stop()
        except Exception as e:
            st.error(f"Error loading code as module: {str(e)}")
            st.stop()
        self.re_download_files()

    @abstractmethod
    def re_download_files(self) -> None:
        """Re-download the files."""
        pass

    def edit_tool(self, tool_name: str) -> None:
        """Edit tool."""
        hf_repo = AgentHub(self.repo_id)
        if tool_name:
            file_path = hf_repo.get_file_path(tool_name, AgentHub.tools_dir)
            with open(file_path) as f:
                default_code = f.read()
        else:
            template_path = os.path.join(os.path.dirname(__file__), "tool_template.py")
            with open(template_path) as f:
                default_code = f.read()

        code = code_editor(
            default_code, lang="python", height=300, options={"wrap": True}
        )

        if not code["text"]:
            st.warning(
                "Not saved. "
                "Press `Control + Enter` (Windows) or `Command + Enter` (Mac) to save."
            )
        tool_name = st.text_input(
            "Tool name",
            value=tool_name,
            disabled=not code["text"],
        )

        button_disabled = not tool_name or not code["text"]
        if st.button(
            "Save",
            disabled=button_disabled,
            on_click=self.save_tool,
            args=(tool_name, code["text"]),
        ):
            st.success(f"Tool `{tool_name}` saved successfully!")

    def run(self) -> None:
        """Main function."""
        st.title("Custom Function or Pydantic Model")
        selected_tool = self.tool_selector(self.repo_id)
        self.edit_tool(selected_tool)
