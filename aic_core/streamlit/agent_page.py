"""Agent page."""

import asyncio
import streamlit as st
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequestPart
from pydantic_ai.messages import ModelResponsePart
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart
from aic_core.agent.agent import AgentConfig
from aic_core.agent.agent import AgentFactory
from aic_core.streamlit.mixins import AgentSelectorMixin
from aic_core.streamlit.page import AICPage
from aic_core.streamlit.page import app_state_registry


class PageState:
    """Page state."""

    chat_history: list[ModelMessage] = []


class AgentPage(AICPage, AgentSelectorMixin):
    """Chatbot page."""

    def __init__(
        self, repo_id: str, page_state: PageState, page_title: str = "Agent"
    ) -> None:
        """Initialize the page."""
        super().__init__()
        self.page_state = page_state
        self.repo_id = repo_id
        self.page_title = page_title
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

    def to_simple_messages(
        self, msg_parts: list[ModelRequestPart] | list[ModelResponsePart]
    ) -> list[tuple[str, str]]:
        """Convert message parts to simple messages."""
        result = []
        for part in msg_parts:
            match part:
                case TextPart():
                    result.append((self.assistant_role, part.content))
                case UserPromptPart():
                    result.append((self.user_role, part.content))  # type: ignore
                case _:  # pragma: no cover
                    pass

        return result

    def display_chat_history(self) -> None:
        """Display chat history."""
        for msg in self.page_state.chat_history:
            simp_msgs = self.to_simple_messages(msg.parts)
            for simp_msg in simp_msgs:
                st.chat_message(simp_msg[0]).write(simp_msg[1])

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
