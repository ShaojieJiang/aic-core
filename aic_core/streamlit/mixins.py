"""Shared components."""

import streamlit as st
from aic_core.agent.agent_hub import AgentHub


class ToolSelectorMixin:
    """Tool selector mixin. A tool is a function or a Pydantic model."""

    def list_tool_names(self, repo_id: str) -> list[str]:
        """List tool names from Tools Hub."""
        hf_repo = AgentHub(repo_id)
        return hf_repo.list_files(AgentHub.tools_dir)

    def tool_selector(self, repo_id: str) -> list[str]:
        """Tool selector."""
        # Get tool names from Tools Hub
        tool_names = self.list_tool_names(repo_id)
        selected_tool_names = st.sidebar.multiselect(
            "Function or Pydantic model name",
            tool_names,
        )
        return selected_tool_names


class AgentSelectorMixin:
    """Agent selector mixin."""

    def list_agent_names(self, repo_id: str) -> list[str]:
        """List all agents."""
        hf_repo = AgentHub(repo_id)
        return hf_repo.list_files(AgentHub.agents_dir)

    def agent_selector(self, repo_id: str) -> str:
        """Agent selector."""
        agent_names = self.list_agent_names(repo_id)
        return st.sidebar.selectbox("Agent", agent_names)
