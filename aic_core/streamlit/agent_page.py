"""Agent page."""

import asyncio
import json
import streamlit as st
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequestPart,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aic_core.agent.agent import AICAgent
from aic_core.agent.result_types import ComponentRegistry
from aic_core.streamlit.mixins import AgentSelectorMixin
from aic_core.streamlit.page import AICPage


class PageState:
    """Page state.

    Acting as a template here. Must be defined in the page that uses it.
    """

    chat_history: list[ModelMessage] = []


class AgentPage(AICPage, AgentSelectorMixin):
    """Agent page.

    PageState needs to have the following values:
    - chat_history: list[ModelMessage]
    """

    def __init__(
        self, repo_id: str, page_state: PageState, page_title: str = "Agent"
    ) -> None:
        """Initialize the page."""
        super().__init__()
        self.repo_id = repo_id
        self.page_title = page_title
        self.page_state = page_state
        self.user_role = "user"
        self.assistant_role = "assistant"
        self.agent: AICAgent | None = None

    def reset_chat_history(self) -> None:
        """Reset chat history."""
        self.page_state.chat_history = []

    async def get_response(self, user_input: str, manual_answer: bool = True) -> None:
        """Get response from agent."""
        history = self.page_state.chat_history
        if manual_answer:  # pragma: no cover
            st.chat_message(self.user_role).write(user_input)
        assert self.agent
        new_messages = await self.agent.get_response(user_input, history)
        self.page_state.chat_history.extend(new_messages)

    def input_callback(
        self, key: str, tool_call_part: ToolCallPart, tool_return_part: ToolReturnPart
    ) -> None:
        """Callback for input components."""
        value = st.session_state[key]
        updated_args = tool_call_part.args_as_dict()
        updated_args.update({"user_input": value})
        tool_call_part.args = (
            updated_args
            if isinstance(tool_call_part.args, dict)
            else json.dumps(updated_args)
        )
        if tool_return_part:  # pragma: no cover
            tool_return_part.content = f"User input: {value}"
        asyncio.run(
            self.get_response(
                f"My answer to '{tool_call_part.args_as_dict().get('label')}' "
                f"is: {value}",
                manual_answer=False,
            )
        )

    def display_parts(
        self,
        msg_parts: list[ModelRequestPart] | list[ModelResponsePart],
        next_msg_part: ModelRequestPart | ModelResponsePart | None,
    ) -> None:
        """Display message parts."""
        for part in msg_parts:
            match part:
                case TextPart():
                    st.chat_message(self.assistant_role).write(part.content)
                case UserPromptPart():
                    st.chat_message(self.user_role).write(part.content)
                case ToolCallPart():
                    if not ComponentRegistry.contains_component(part.tool_name):
                        return

                    assert isinstance(next_msg_part, ToolReturnPart)
                    with st.chat_message(self.assistant_role):
                        ComponentRegistry.generate_st_component(
                            part, next_msg_part, self.input_callback
                        )
                case _:  # pragma: no cover
                    pass

    def display_chat_history(self) -> None:
        """Display chat history."""
        history_shifted = self.page_state.chat_history[1:] + [None]
        for msg, next_msg in zip(
            self.page_state.chat_history, history_shifted, strict=False
        ):
            self.display_parts(msg.parts, None if not next_msg else next_msg.parts[0])

    def run(self) -> None:
        """Run the page."""
        st.title(self.page_title)
        self.display_chat_history()

        agent_name = self.agent_selector(self.repo_id)
        self.agent = AICAgent(self.repo_id, agent_name)
        st.sidebar.button("Reset chat history", on_click=self.reset_chat_history)
        user_input = st.chat_input("Enter a message")

        if user_input:  # pragma: no cover
            asyncio.run(self.get_response(user_input))
            st.rerun()
