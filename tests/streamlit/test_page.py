import pytest
from streamlit import session_state
from aic_core.streamlit.page import AICPage
from aic_core.streamlit.page import PageStateSingleton


def test_page_state_singleton_creation():
    # Clear session state before test
    if "test_file.py" in session_state:
        del session_state["test_file.py"]

    # Create first instance
    state1 = PageStateSingleton("test_file.py")
    assert state1.file_path == "test_file.py"

    # Create second instance with same file_path
    state2 = PageStateSingleton("test_file.py")
    assert state1 is state2  # Should be the same instance

    # Create instance with different file_path
    state3 = PageStateSingleton("other_file.py")
    assert state1 is not state3  # Should be different instances


class TestAICPage(AICPage):
    def run(self) -> None:
        pass


def test_aic_page_abstract():
    # Should be able to instantiate concrete implementation
    page = TestAICPage()
    assert isinstance(page, AICPage)

    # Should not be able to instantiate abstract class
    with pytest.raises(TypeError):
        AICPage()
