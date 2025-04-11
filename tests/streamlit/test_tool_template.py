import pytest
from aic_core.streamlit.tool_template import MyModel, my_function


def test_my_function():
    # Test basic addition
    assert my_function(2, 3) == 5
    assert my_function(0, 0) == 0
    assert my_function(-1, 1) == 0

    # Test with larger numbers
    assert my_function(100, 200) == 300


def test_my_model():
    # Test valid model creation
    model = MyModel(attr1="test", attr2=42)
    assert model.attr1 == "test"
    assert model.attr2 == 42

    # Test model validation errors
    with pytest.raises(ValueError):
        MyModel(attr1=None, attr2=42)  # attr1 cannot be None

    with pytest.raises(ValueError):
        MyModel(attr1="test", attr2="not_an_integer")  # attr2 must be an integer
