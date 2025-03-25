"""Agent config page."""

from typing import get_args
import streamlit as st
from pydantic_ai.models import KnownModelName
from aic_core.agent.agent import AgentConfig
from aic_core.agent.agent_hub import AgentHub
from aic_core.streamlit.mixins import AgentSelectorMixin
from aic_core.streamlit.mixins import ToolSelectorMixin
from aic_core.streamlit.page import AICPage


class AgentConfigPage(AICPage, AgentSelectorMixin, ToolSelectorMixin):
    """Agent config page."""

    def __init__(self, repo_id: str) -> None:
        """Initialize the page."""
        super().__init__()
        self.repo_id = repo_id

    def list_result_type_options(self) -> list[str]:
        """List all result types."""
        primitive_types = ["str", "int", "float", "bool"]
        hf_repo = AgentHub(self.repo_id)
        return primitive_types + hf_repo.list_files(AgentHub.pydantic_models_dir)

    def configure(self, config: AgentConfig) -> AgentConfig:
        """Widgets to configure the agent."""
        model_options = [
            model for model in get_args(KnownModelName) if model.startswith("openai")
        ]
        model = st.selectbox(
            "Select a model",
            model_options,
            index=model_options.index(config.model),
        )

        result_type_options = self.list_result_type_options()
        result_type = st.multiselect(
            "Result type",
            options=result_type_options,
            default=config.result_type,
        )
        system_prompt = st.text_area("System prompt", value=config.system_prompt)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config.model_settings.get("temperature", 1.0)
            if config.model_settings
            else 1.0,
        )
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=config.model_settings.get("top_p", 1.0)
            if config.model_settings
            else 1.0,
        )
        model_settings = {
            "temperature": temperature,
            "top_p": top_p,
        }
        retries = st.number_input(
            "Retries", min_value=0, max_value=100, value=config.retries
        )
        result_tool_name = st.text_input(
            "Result tool name", value=config.result_tool_name
        )
        result_tool_description = st.text_area(
            "Result tool description", value=config.result_tool_description
        )
        result_retries = st.number_input(
            "Result retries", min_value=0, max_value=100, value=config.retries
        )
        list_known_tools = self.list_tool_names(self.repo_id)
        default_known_tools = [
            tool for tool in config.known_tools if tool in list_known_tools
        ]
        known_tools = st.multiselect(
            "Known tools", options=list_known_tools, default=default_known_tools
        )
        hf_tools = st.text_area("HF tools", value="\n".join(config.hf_tools))
        mcp_servers = st.text_area("MCP servers", value="\n".join(config.mcp_servers))
        defer_model_check = st.toggle(
            "Defer model check", value=config.defer_model_check
        )
        end_strategy = st.selectbox("End strategy", ["early", "exhaustive"])
        name = st.text_input("Name", value=config.name)

        return AgentConfig(
            model=model,
            result_type=result_type,
            system_prompt=system_prompt,
            model_settings=model_settings,
            retries=retries,
            result_tool_name=result_tool_name,
            result_tool_description=result_tool_description,
            result_retries=result_retries,
            known_tools=known_tools,
            hf_tools=[x for x in hf_tools.split("\n") if x],
            mcp_servers=[x for x in mcp_servers.split("\n") if x],
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,
            name=name,
            repo_id=self.repo_id,
        )

    def save_config(self, config: AgentConfig) -> None:
        """Save the config."""
        config.push_to_hub()
        AgentHub(self.repo_id).download_files()

    def run(self) -> None:
        """Main function."""
        st.title("Custom Agent")
        agent_name = self.agent_selector(self.repo_id)
        if agent_name:
            config = AgentConfig.from_hub(self.repo_id, agent_name)
        else:  # Initialize a new agent
            config = AgentConfig(model="openai:gpt-4o", repo_id=self.repo_id)
        new_config = self.configure(config)
        if st.button("Save", on_click=self.save_config, args=(new_config,)):
            st.success("Agent pushed to the hub.")
