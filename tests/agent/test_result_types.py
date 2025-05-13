import json
from unittest.mock import patch
import pytest
from pydantic_ai.messages import ToolCallPart
from aic_core.agent.result_types import (
    Choice,
    ComponentRegistry,
    JsonOutput,
    NumberInput,
    TableOutput,
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
        patch("streamlit.dataframe") as mock_table,
        patch("streamlit.form") as mock_form,
        patch("streamlit.form_submit_button") as mock_form_submit,
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
        mock_form.return_value.__enter__ = lambda x: None
        mock_form.return_value.__exit__ = lambda x, y, z, w: None

        yield {
            "text_input": mock_text_input,
            "number_input": mock_number_input,
            "radio": mock_radio,
            "multiselect": mock_multiselect,
            "text": mock_text,
            "markdown": mock_markdown,
            "latex": mock_latex,
            "json": mock_json,
            "dataframe": mock_table,
            "form": mock_form,
            "form_submit_button": mock_form_submit,
        }


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
    part = ToolCallPart(
        tool_name="final_result_NumberInput",
        args=json.dumps(params.model_dump()),
        tool_call_id="call_test",
        part_kind="tool-call",
    )

    ComponentRegistry.generate_st_component(part)

    mock_streamlit["form"].assert_called_once_with(key="test_key")
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

    part = ToolCallPart(
        tool_name="final_result_Choice",
        args=json.dumps(params.model_dump()),
        tool_call_id="call_test",
        part_kind="tool-call",
    )
    ComponentRegistry.generate_st_component(part)

    mock_streamlit["form"].assert_called_once_with(key="test_key")
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

    part = ToolCallPart(
        tool_name="final_result_Choice",
        args=json.dumps(params.model_dump()),
        tool_call_id="call_test",
        part_kind="tool-call",
    )
    ComponentRegistry.generate_st_component(part)

    mock_streamlit["form"].assert_called_once_with(key="test_key")
    mock_streamlit["multiselect"].assert_called_once()
    call_args = mock_streamlit["multiselect"].call_args[1]
    assert call_args["label"] == "Test Multiselect"
    assert call_args["key"] == "test_key"
    assert call_args["options"] == ["option1", "option2", "option3"]
    assert call_args["default"] == ["option1", "option2"]


def test_generate_latex_output(mock_streamlit):
    """Test generating a latex output component."""
    params = JsonOutput(type="json", body="Test output latex")

    part = ToolCallPart(
        tool_name="final_result_JsonOutput",
        args=json.dumps(params.model_dump()),
        tool_call_id="call_test",
        part_kind="tool-call",
    )
    ComponentRegistry.generate_st_component(part)

    mock_streamlit["form"].assert_called_once_with(key=None)
    mock_streamlit["json"].assert_called_once_with("Test output latex")
    mock_streamlit["form_submit_button"].assert_called_once()


def test_generate_table_output(mock_streamlit):
    """Test generating a table output component."""
    data = [
        {"column1": 1, "column2": "a"},
        {"column1": 2, "column2": "b"},
        {"column1": 3, "column2": "c"},
    ]
    params = TableOutput(type="dataframe", data=data)

    part = ToolCallPart(
        tool_name="final_result_TableOutput",
        args=json.dumps(params.model_dump()),
        tool_call_id="call_test",
        part_kind="tool-call",
    )
    ComponentRegistry.generate_st_component(part)

    mock_streamlit["form"].assert_called_once_with(key=None)
    mock_streamlit["dataframe"].assert_called_once()
    call_args = mock_streamlit["dataframe"].call_args[1]
    assert call_args["data"] == data
    mock_streamlit["form_submit_button"].assert_called_once()


def test_generate_json_output(mock_streamlit):
    """Test generating a json output component."""
    test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    params = JsonOutput(type="json", body=test_data)

    part = ToolCallPart(
        tool_name="final_result_JsonOutput",
        args=json.dumps(params.model_dump()),
        tool_call_id="call_test",
        part_kind="tool-call",
    )
    ComponentRegistry.generate_st_component(part)

    mock_streamlit["form"].assert_called_once_with(key=None)
    mock_streamlit["json"].assert_called_once_with(test_data)
    mock_streamlit["form_submit_button"].assert_called_once()


def test_contains_component():
    """Test checking if a component is registered."""
    # Test with registered component without prefix
    assert ComponentRegistry.contains_component("JsonOutput") is True
    # Test with registered component with prefix
    assert ComponentRegistry.contains_component("final_result_JsonOutput") is True
    # Test with unregistered component
    assert ComponentRegistry.contains_component("UnknownComponent") is False
