from unittest.mock import Mock, patch
import pytest
from aic_core.agent.agent import AgentConfig
from aic_core.agent.agent_hub import AgentHub
from aic_core.streamlit.agent_config import AgentConfigPage


@pytest.fixture
def mock_streamlit():
    """Mock streamlit components."""
    with (
        patch("aic_core.streamlit.agent_config.st") as mock_st,
        patch("aic_core.streamlit.agent_config.code_editor") as mock_code_editor,
    ):
        # Configure default mock returns for streamlit widgets
        mock_st.selectbox.return_value = "openai:gpt-4"
        mock_st.multiselect.return_value = ["str"]
        mock_code_editor.return_value = {"text": "test prompt"}
        mock_st.slider.return_value = 1.0
        mock_st.number_input.return_value = 3
        mock_st.text_input.return_value = "test_name"
        mock_st.toggle.return_value = False
        yield mock_st


@pytest.fixture
def agent_config_page():
    """Create a concrete AgentConfigPage instance for testing."""

    class ConcreteAgentConfigPage(AgentConfigPage):
        def re_download_files(self) -> None:
            pass

    return ConcreteAgentConfigPage("test-repo")


def test_list_result_type_options(agent_config_page):
    """Test listing result type options."""
    with patch.object(
        agent_config_page, "list_result_type_names", return_value=["CustomType"]
    ):
        result = agent_config_page.list_result_type_options()
        assert result == ["str", "int", "float", "bool", "CustomType"]


def test_configure(agent_config_page, mock_streamlit):
    """Test agent configuration."""
    initial_config = AgentConfig(model="openai:gpt-4", repo_id="test-repo")
    with (
        patch.object(
            agent_config_page, "list_result_type_names", return_value=["CustomType"]
        ),
        patch.object(
            agent_config_page,
            "list_function_names",
            return_value=["function1", "function2"],
        ),
    ):
        result = agent_config_page.configure(initial_config)

        assert isinstance(result, AgentConfig)
        assert result.model == "openai:gpt-4"
        assert result.result_type == ["str"]
        assert result.system_prompt == "test prompt"
        assert result.model_settings == {"temperature": 1.0, "top_p": 1.0}
        assert result.retries == 3
        assert result.name == "test_name"
        assert result.repo_id == "test-repo"


def test_save_config(agent_config_page):
    """Test saving configuration."""
    mock_config = Mock(spec=AgentConfig)
    with patch.object(AgentHub, "download_files"):
        agent_config_page.save_config(mock_config)

    mock_config.push_to_hub.assert_called_once()


def test_run(agent_config_page, mock_streamlit):
    """Test the run method."""
    with (
        patch.object(agent_config_page, "agent_selector", return_value="test_agent"),
        patch.object(AgentConfig, "from_hub") as mock_from_hub,
        patch.object(agent_config_page, "save_config"),
    ):
        mock_config = Mock(spec=AgentConfig)
        mock_from_hub.return_value = mock_config
        with patch.object(agent_config_page, "configure", return_value=mock_config):
            agent_config_page.run()

            mock_streamlit.title.assert_called_once_with("Custom Agent")
            agent_config_page.agent_selector.assert_called_once_with("test-repo")
            mock_from_hub.assert_called_once_with("test-repo", "test_agent")


def test_run_new_agent(agent_config_page, mock_streamlit):
    """Test the run method when creating a new agent."""
    with (
        patch.object(agent_config_page, "agent_selector", return_value=None),
        patch.object(agent_config_page, "save_config"),
    ):
        mock_config = Mock(spec=AgentConfig)
        with patch.object(agent_config_page, "configure", return_value=mock_config):
            agent_config_page.run()

            mock_streamlit.title.assert_called_once_with("Custom Agent")
            agent_config_page.agent_selector.assert_called_once_with("test-repo")
