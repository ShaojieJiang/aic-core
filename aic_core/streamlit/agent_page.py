"""Agent page."""

import asyncio
import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequestPart,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aic_core.agent.agent import AgentConfig, AgentFactory
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

    def reset_chat_history(self) -> None:
        """Reset chat history."""
        self.page_state.chat_history = []

    def get_agent(self, agent_name: str) -> Agent:
        """Get agent."""
        agent_config = AgentConfig.from_hub(self.repo_id, agent_name)
        agent_factory = AgentFactory(agent_config)
        agent = agent_factory.create_agent()

        return agent

    async def get_response(self, user_input: str, agent: Agent) -> None:
        """Get response from agent."""
        history = self.page_state.chat_history
        st.chat_message(self.user_role).write(user_input)
        if agent._mcp_servers:
            async with agent.run_mcp_servers():
                result = await agent.run(user_input, message_history=history)  # type: ignore
        else:
            result = await agent.run(user_input, message_history=history)  # type: ignore
        self.page_state.chat_history.extend(result.new_messages())

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
                        ComponentRegistry.generate_st_component(part, next_msg_part)
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
        agent = self.get_agent(agent_name)
        st.sidebar.button("Reset chat history", on_click=self.reset_chat_history)
        user_input = st.chat_input("Enter a message")

        if user_input:  # pragma: no cover
            asyncio.run(self.get_response(user_input, agent))
            st.rerun()
