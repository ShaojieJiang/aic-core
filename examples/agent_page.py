"""Chatbot page."""

import os
from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage
from aic_core.streamlit.agent_page import AgentPage, PageState
from aic_core.streamlit.page import app_state


load_dotenv()


@app_state(__file__)
class ChatbotState(PageState):
    """Chatbot state."""

    chat_history: list[ModelMessage] = []


AgentPage(os.environ["HF_REPO_ID"], ChatbotState(), "Extendable Agents").run()
