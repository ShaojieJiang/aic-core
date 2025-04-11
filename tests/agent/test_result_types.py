from unittest.mock import patch
import pytest
from aic_core.agent.result_types import (
    Choice,
    ComponentRegistry,
    NumberInput,
    TableOutput,
    TextInput,
    TextOutput,
)


@pytest.fixture
def mock_streamlit():
    """Fixture to mock streamlit components."""
    with (
        patch("streamlit.text_input") as mock_text_input,
        patch("streamlit.number_input") as mock_number_input,
        patch("streamlit.radio") as mock_radio,
        patch("streamlit.multiselect") as mock_multiselect,
        patch("streamlit.text") as mock_text,
        patch("streamlit.markdown") as mock_markdown,
        patch("streamlit.latex") as mock_latex,
        patch("streamlit.json") as mock_json,
        patch("streamlit.table") as mock_table,
    ):
        # Setup return values for mocks
        mock_text_input.return_value = "test input"
        mock_number_input.return_value = 42
        mock_radio.return_value = "option1"
        mock_multiselect.return_value = ["option1", "option2"]
        mock_text.return_value = None
        mock_markdown.return_value = None
        mock_latex.return_value = None
        mock_json.return_value = None
        mock_table.return_value = None

        yield {
            "text_input": mock_text_input,
            "number_input": mock_number_input,
            "radio": mock_radio,
            "multiselect": mock_multiselect,
            "text": mock_text,
            "markdown": mock_markdown,
            "latex": mock_latex,
            "json": mock_json,
            "table": mock_table,
        }


def test_generate_text_input(mock_streamlit):
    """Test generating a text input component."""
    params = TextInput(
        type="text_input",
        label="Test Input",
        key="test_key",
        help="Test help",
        user_input="initial value",
    )

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["text_input"].assert_called_once()
    call_args = mock_streamlit["text_input"].call_args[1]
    assert call_args["label"] == "Test Input"
    assert call_args["key"] == "test_key"
    assert call_args["help"] == "Test help"
    assert call_args["value"] == "initial value"


def test_generate_number_input(mock_streamlit):
    """Test generating a number input component."""
    params = NumberInput(
        type="number_input",
        label="Test Number",
        key="test_key",
        min_value=0,
        max_value=100,
        step=1,
        user_input=42,
    )

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["number_input"].assert_called_once()
    call_args = mock_streamlit["number_input"].call_args[1]
    assert call_args["label"] == "Test Number"
    assert call_args["key"] == "test_key"
    assert call_args["min_value"] == 0
    assert call_args["max_value"] == 100
    assert call_args["step"] == 1
    assert call_args["value"] == 42


def test_generate_radio(mock_streamlit):
    """Test generating a radio component."""
    params = Choice(
        type="radio",
        label="Test Radio",
        key="test_key",
        options=["option1", "option2", "option3"],
        user_input="option1",
    )

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["radio"].assert_called_once()
    call_args = mock_streamlit["radio"].call_args[1]
    assert call_args["label"] == "Test Radio"
    assert call_args["key"] == "test_key"
    assert call_args["options"] == ["option1", "option2", "option3"]
    assert call_args["index"] == 0  # index of "option1"


def test_generate_multiselect(mock_streamlit):
    """Test generating a multiselect component."""
    params = Choice(
        type="multiselect",
        label="Test Multiselect",
        key="test_key",
        options=["option1", "option2", "option3"],
        user_input=["option1", "option2"],
    )

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["multiselect"].assert_called_once()
    call_args = mock_streamlit["multiselect"].call_args[1]
    assert call_args["label"] == "Test Multiselect"
    assert call_args["key"] == "test_key"
    assert call_args["options"] == ["option1", "option2", "option3"]
    assert call_args["default"] == ["option1", "option2"]


def test_generate_text_output(mock_streamlit):
    """Test generating a text output component."""
    params = TextOutput(type="text", body="Test output text")

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["text"].assert_called_once_with("Test output text")


def test_generate_markdown_output(mock_streamlit):
    """Test generating a markdown output component."""
    params = TextOutput(type="markdown", body="# Test Markdown")

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["markdown"].assert_called_once_with("# Test Markdown")


def test_generate_table_output(mock_streamlit):
    """Test generating a table output component."""
    params = TableOutput(
        type="table", data={"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
    )

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["table"].assert_called_once()
    call_args = mock_streamlit["table"].call_args[1]
    assert call_args["data"] == {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}


def test_generate_json_output(mock_streamlit):
    """Test generating a json output component."""
    params = TextOutput(
        type="json", body={"key": "value", "number": 42, "list": [1, 2, 3]}
    )

    ComponentRegistry.generate_st_component(params)

    mock_streamlit["json"].assert_called_once_with(
        {"key": "value", "number": 42, "list": [1, 2, 3]}
    )
