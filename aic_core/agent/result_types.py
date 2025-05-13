"""Streamlit MCP server with Pydantic models."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Literal
import streamlit as st
from pydantic import BaseModel
from pydantic_ai.messages import ToolCallPart, ToolReturnPart


class ComponentRegistry:
    """Registry for Streamlit component models."""

    _registry: dict[str, type[InputComponent | OutputComponent]] = {}

    @classmethod
    def register(cls) -> Callable:
        """Decorator to register a component model.

        Returns:
            A decorator function that registers the component model using its
            class name as the key
        """

        def decorator(
            component_class: type[InputComponent | OutputComponent],
        ) -> type[InputComponent | OutputComponent]:
            cls._registry[component_class.__name__] = component_class
            return component_class

        return decorator

    @classmethod
    def get_component_class(
        cls, component_name: str
    ) -> type[InputComponent | OutputComponent]:
        """Get a component class by its name.

        Args:
            component_name: The name of the component class

        Returns:
            The component class

        Raises:
            KeyError: If the component is not registered
        """
        if component_name not in cls._registry:  # pragma: no cover
            raise KeyError(f"Component '{component_name}' is not registered")
        return cls._registry[component_name]

    @classmethod
    def get_registered_components(cls) -> list[str]:
        """Get a list of all registered component names.

        Returns:
            A list of registered component class names
        """
        return list(cls._registry.keys())

    @classmethod
    def contains_component(cls, tool_name: str) -> bool:
        """Check if a component is registered."""
        return tool_name.replace("final_result_", "") in cls._registry

    @classmethod
    def generate_st_component(
        cls,
        tool_call_part: ToolCallPart,
        tool_return_part: ToolReturnPart | None = None,
        input_callback: Callable | None = None,
    ) -> Any:
        """Generate a component based on the parameters."""
        class_name = tool_call_part.tool_name.replace("final_result_", "")
        model = ComponentRegistry.get_component_class(class_name)
        params = model.model_validate(tool_call_part.args_as_dict())

        comp_type = params.type
        comp_func = getattr(st, comp_type)
        kwargs = params.model_dump(exclude={"type"})
        value = kwargs.pop("user_input", None)
        key = kwargs.get("key", None)
        with st.form(key=key):
            match comp_type:
                case "text_input" | "text_area" | "number_input" | "slider":
                    output = comp_func(
                        **kwargs,
                        value=value,
                    )
                case "radio":
                    try:
                        index = kwargs["options"].index(value)
                    except ValueError:  # pragma: no cover
                        index = None
                    output = comp_func(
                        **kwargs,
                        index=index,
                    )
                case "multiselect":
                    output = comp_func(
                        **kwargs,
                        default=value,
                    )
                case "latex":  # pragma: no cover
                    output = comp_func(kwargs["body"])
                case "json":
                    output = comp_func(kwargs["body"])
                case _:
                    output = comp_func(**kwargs)

            if st.form_submit_button(
                "Submit",
                on_click=input_callback,
                args=(key, tool_call_part, tool_return_part),
            ):  # pragma: no cover
                pass
        return output


class StreamlitComponent(BaseModel, use_attribute_docstrings=True):
    """Parameters for components."""

    type: str
    """Streamlit component type."""


class InputComponent(StreamlitComponent):
    """Parameters for input components."""

    label: str
    """Label for the component."""
    key: str
    """Unique key for the component."""
    help: str | None = None
    """Help text for the component."""
    user_input: Any | None = None
    """Value input by the user for the component."""


class OutputComponent(StreamlitComponent):
    """Parameters for output components."""

    pass


@ComponentRegistry.register()
class NumberInput(InputComponent):
    """Parameters for number input components."""

    type: Literal["number_input", "slider"]
    """Streamlit component type."""
    min_value: int | float | None = None
    """Minimum value for the component."""
    max_value: int | float | None = None
    """Maximum value for the component."""
    step: int | float | None = None
    """Step for the component."""
    user_input: int | float | None = None
    """Value input by the user for the component."""


@ComponentRegistry.register()
class Choice(InputComponent):
    """Parameters for choice components."""

    type: Literal["radio", "multiselect"]
    """Streamlit component type."""
    options: list[str]
    """Options for the component."""
    user_input: int | str | list[str] | None = None
    """Value input by the user for the component."""


@ComponentRegistry.register()
class JsonOutput(OutputComponent):
    """Parameters for JSON output components."""

    type: Literal["json"]
    """Streamlit component type."""
    body: str | dict
    """Body for the component."""


@ComponentRegistry.register()
class TableOutput(OutputComponent):
    """Parameters for table output components."""

    type: str = "dataframe"
    """Streamlit component type."""
    data: list[dict]
    """List of dictionaries for the component."""
