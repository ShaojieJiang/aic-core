from unittest.mock import Mock
from unittest.mock import mock_open
from unittest.mock import patch
import pytest
from aic_core.streamlit.tool_config import ToolConfigPage


@pytest.fixture
def tool_config():
    class TestToolConfigPage(ToolConfigPage):
        def re_download_files(self):
            pass

    return TestToolConfigPage("test-repo")


def test_load_code_as_module(tool_config):
    code = """
def test_function():
    return "Hello"
    
class TestClass:
    pass
"""
    module = tool_config.load_code_as_module(code)
    assert hasattr(module, "test_function")
    assert hasattr(module, "TestClass")
    assert module.test_function() == "Hello"


@pytest.mark.parametrize(
    "tool_name,code,is_model",
    [
        ("test_tool", "def test_tool(): pass", False),
        (
            "TestModel",
            "from pydantic import BaseModel\nclass TestModel(BaseModel):\n    field: str",
            True,
        ),
    ],
)
def test_save_tool(tool_config, tool_name, code, is_model):
    mock_hub = Mock()
    with patch("aic_core.streamlit.tool_config.AgentHub", return_value=mock_hub):
        tool_config.save_tool(tool_name, code)


def test_save_tool_invalid_definition(tool_config):
    with patch("streamlit.error") as mock_error, patch("streamlit.stop") as mock_stop:
        tool_config.save_tool("missing_tool", "def other_tool(): pass")
        mock_error.assert_called_once_with(
            "Definition `missing_tool` not found in module"
        )
        mock_stop.assert_called_once()


def test_edit_tool_existing(tool_config):
    mock_content = "def existing_tool(): pass"
    mock_hub = Mock()
    mock_hub.get_file_path.return_value = "fake/path"

    with (
        patch("aic_core.streamlit.tool_config.AgentHub", return_value=mock_hub),
        patch("builtins.open", mock_open(read_data=mock_content)),
        patch("aic_core.streamlit.tool_config.code_editor") as mock_editor,
        patch("streamlit.text_input") as mock_input,
        patch("streamlit.button") as mock_button,
    ):
        mock_editor.return_value = {"text": mock_content}
        mock_input.return_value = "existing_tool"
        mock_button.return_value = True

        tool_config.edit_tool("existing_tool")
        mock_hub.get_file_path.assert_called_once()
        mock_editor.assert_called_once()
        mock_input.assert_called_once()


def test_edit_tool_new(tool_config):
    mock_template = "# Template content"

    with (
        patch("builtins.open", mock_open(read_data=mock_template)),
        patch("aic_core.streamlit.tool_config.code_editor") as mock_editor,
        patch("streamlit.text_input") as mock_input,
        patch("streamlit.button") as mock_button,
    ):
        mock_editor.return_value = {"text": mock_template}
        mock_input.return_value = "new_tool"
        mock_button.return_value = True

        tool_config.edit_tool("")
        mock_editor.assert_called_once()
        mock_input.assert_called_once()


def test_run(tool_config):
    with (
        patch("streamlit.title") as mock_title,
        patch.object(
            tool_config, "tool_selector", return_value="test_tool"
        ) as mock_selector,
        patch.object(tool_config, "edit_tool") as mock_edit,
    ):
        tool_config.run()

        # Verify the title is set correctly
        mock_title.assert_called_once_with("Custom Function or Pydantic Model")

        # Verify tool_selector was called with repo_id
        mock_selector.assert_called_once_with(tool_config.repo_id)

        # Verify edit_tool was called with selected tool
        mock_edit.assert_called_once_with("test_tool")
