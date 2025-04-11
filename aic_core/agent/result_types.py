"""Streamlit MCP server with Pydantic models."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Literal
import streamlit as st
from pydantic import BaseModel


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
    def generate_st_component(cls, params: InputComponent | OutputComponent) -> Any:
        """Generate a component based on the parameters."""

        def _input_callback(
            key: str, params: InputComponent
        ) -> None:  # pragma: no cover
            value = st.session_state[key]
            params.user_input = value

        comp_type = params.type
        comp_func = getattr(st, comp_type)
        kwargs = params.model_dump(exclude={"type"})
        value = kwargs.pop("user_input", None)
        key = kwargs.get("key", None)
        match comp_type:
            case "text_input" | "text_area" | "number_input" | "slider":
                output = comp_func(
                    **kwargs, value=value, on_change=_input_callback, args=(key, params)
                )
            case "radio":
                try:
                    index = kwargs["options"].index(value)
                except ValueError:  # pragma: no cover
                    index = None
                output = comp_func(
                    **kwargs, index=index, on_change=_input_callback, args=(key, params)
                )
            case "multiselect":
                output = comp_func(
                    **kwargs,
                    default=value,
                    on_change=_input_callback,
                    args=(key, params),
                )
            case "text" | "markdown" | "latex":
                output = comp_func(kwargs["body"])
            case "json":
                output = comp_func(kwargs["body"])
            case _:
                output = comp_func(**kwargs)
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
class TextInput(InputComponent):
    """Parameters for text input components."""

    type: Literal["text_input", "text_area"]
    """Streamlit component type."""
    user_input: str = ""
    """Value input by the user for the component."""


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
class TextOutput(OutputComponent):
    """Parameters for text output components."""

    type: Literal["text", "markdown", "latex", "code", "json"]
    """Streamlit component type."""
    body: str | dict
    """Body for the component."""
    language: str = "python"
    """Language for the component."""


@ComponentRegistry.register()
class TableOutput(OutputComponent):
    """Parameters for table output components."""

    type: Literal["table", "dataframe"]
    """Streamlit component type."""
    data: dict
    """Data dict for the component."""
