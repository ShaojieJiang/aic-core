from unittest.mock import Mock
from unittest.mock import patch
import pytest
from aic_core.streamlit.mixins import AgentSelectorMixin
from aic_core.streamlit.mixins import ToolSelectorMixin


@pytest.fixture
def tool_selector():
    return ToolSelectorMixin()


@pytest.fixture
def agent_selector():
    return AgentSelectorMixin()


@pytest.fixture
def mock_agent_hub():
    with patch("aic_core.streamlit.mixins.AgentHub") as mock:
        # Configure the mock to return predictable values
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


def test_list_function_names(tool_selector, mock_agent_hub):
    mock_agent_hub.tools_dir = "tools"
    mock_agent_hub.list_files.return_value = ["function1.py", "function2.py"]

    result = tool_selector.list_function_names("test-repo")

    assert result == ["function1.py", "function2.py"]


def test_list_result_type_names(tool_selector, mock_agent_hub):
    mock_agent_hub.result_types_dir = "result_types"
    mock_agent_hub.list_files.return_value = ["type1.py", "type2.py"]

    result = tool_selector.list_result_type_names("test-repo")

    assert result == ["type1.py", "type2.py"]


def test_list_tool_output_names(tool_selector, mock_agent_hub):
    mock_agent_hub.tools_dir = "tools"
    mock_agent_hub.result_types_dir = "result_types"
    mock_agent_hub.list_files.side_effect = [
        ["function1.py", "function2.py"],
        ["type1.py", "type2.py"],
    ]

    result = tool_selector._list_tool_output_names("test-repo")

    assert result == ["function1.py", "function2.py", "type1.py", "type2.py"]


@patch("streamlit.sidebar.multiselect")
def test_tool_selector(mock_multiselect, tool_selector, mock_agent_hub):
    # Mock the tool names
    mock_agent_hub.list_files.side_effect = [["function1.py"], ["type1.py"]]

    # Mock the multiselect to return a selection
    mock_multiselect.return_value = ["function1.py"]

    result = tool_selector.tool_selector("test-repo")

    mock_multiselect.assert_called_once_with(
        "Function or Pydantic model name",
        ["function1.py", "type1.py"],
        max_selections=1,
    )
    assert result == "function1.py"


@patch("streamlit.sidebar.multiselect")
def test_tool_selector_no_selection(mock_multiselect, tool_selector, mock_agent_hub):
    # Mock the tool names
    mock_agent_hub.list_files.side_effect = [["function1.py"], ["type1.py"]]

    # Mock the multiselect to return no selection
    mock_multiselect.return_value = []

    result = tool_selector.tool_selector("test-repo")

    assert result == ""


def test_list_agent_names(agent_selector, mock_agent_hub):
    mock_agent_hub.agents_dir = "agents"
    mock_agent_hub.list_files.return_value = ["agent1.py", "agent2.py"]

    result = agent_selector.list_agent_names("test-repo")

    assert result == ["agent1.py", "agent2.py"]


@patch("streamlit.sidebar.selectbox")
def test_agent_selector(mock_selectbox, agent_selector, mock_agent_hub):
    mock_agent_hub.agents_dir = "agents"
    mock_agent_hub.list_files.return_value = ["agent1.py", "agent2.py"]
    mock_selectbox.return_value = "agent1.py"

    result = agent_selector.agent_selector("test-repo")

    mock_selectbox.assert_called_once_with("Agent", ["agent1.py", "agent2.py"])
    assert result == "agent1.py"
