import pytest
from streamlit import session_state
from aic_core.streamlit.page import AICPage, app_state


def test_app_state_decorator():
    # Clear session state before test
    session_state.clear()

    # Create a test class to decorate
    @app_state("test_path")
    class TestClass:
        def __init__(self, value=None):
            self.value = value

    # First instance creation
    instance1 = TestClass(value="test")
    assert instance1.value == "test"
    assert "test_path" in session_state
    assert session_state["test_path"] is instance1

    # Second call should return the same instance
    instance2 = TestClass(value="different")
    assert instance2 is instance1
    assert instance2.value == "test"  # Should keep original value

    # Test with different path
    @app_state("other_path")
    class OtherClass:
        def __init__(self, value=None):
            self.value = value

    # Should create new instance for different path
    other_instance = OtherClass(value="other")
    assert "other_path" in session_state
    assert other_instance is not instance1
    assert other_instance.value == "other"


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
