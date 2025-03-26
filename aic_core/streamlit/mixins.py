"""Shared components."""

import streamlit as st
from aic_core.agent.agent_hub import AgentHub


class ToolSelectorMixin:
    """Tool selector mixin. A tool is a function or a Pydantic model."""

    def list_function_names(self, repo_id: str) -> list[str]:
        """List function names from Agents Hub."""
        hf_repo = AgentHub(repo_id)
        return hf_repo.list_files(AgentHub.tools_dir)

    def list_result_type_names(self, repo_id: str) -> list[str]:
        """List result type names from Agents Hub."""
        hf_repo = AgentHub(repo_id)
        return hf_repo.list_files(AgentHub.result_types_dir)

    def _list_tool_output_names(self, repo_id: str) -> list[str]:
        """List tool or output names from Agents Hub."""
        return self.list_function_names(repo_id) + self.list_result_type_names(repo_id)

    def tool_selector(self, repo_id: str) -> str:
        """Tool selector."""
        # Get tool names from Agents Hub
        tool_names = self._list_tool_output_names(repo_id)
        selected_tool_names = st.sidebar.multiselect(
            "Function or Pydantic model name",
            tool_names,
            max_selections=1,
        )
        selected_tool = selected_tool_names[0] if selected_tool_names else ""
        return selected_tool


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
