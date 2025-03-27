import time
from collections.abc import Callable
from unittest.mock import Mock
from unittest.mock import mock_open
from unittest.mock import patch
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import BaseModel
from aic_core.agent.agent_hub import AgentHub


# Test fixtures and helper classes
class DummyModel(BaseModel):
    name: str
    value: int


def dummy_function():
    pass


@pytest.fixture
def agent_hub():
    return AgentHub("test-repo")


# Tests
def test_init():
    repo = AgentHub("test-repo")
    assert repo.repo_id == "test-repo"


@patch("aic_core.agent.agent_hub.snapshot_download")
def test_load_files(mock_snapshot):
    repo = AgentHub("test-repo")
    repo.download_files()
    mock_snapshot.assert_called_once_with(repo_id="test-repo", repo_type="space")


@patch("aic_core.agent.agent_hub.hf_hub_download")
def test_load_config(mock_download):
    repo = AgentHub("test-repo")
    mock_download.return_value = "config.json"

    with (
        patch("builtins.open", mock_open(read_data='{"key": "value"}')),
        patch("aic_core.agent.agent_hub.AgentHub._layzy_update"),
    ):
        result = repo.load_config("config")
        assert result == {"key": "value"}


@patch("aic_core.agent.agent_hub.hf_hub_download")
@patch("aic_core.agent.agent_hub.importlib.util")
def test_load_tool(mock_importlib, mock_download):
    repo = AgentHub("test-repo")
    mock_download.return_value = "/path/to/tool.py"

    # Setup mock module
    mock_module = Mock()
    mock_module.tool = dummy_function
    mock_spec = Mock()
    mock_spec.loader = Mock()

    mock_importlib.spec_from_file_location.return_value = mock_spec
    mock_importlib.module_from_spec.return_value = mock_module

    with (
        patch("aic_core.agent.agent_hub.AgentHub._layzy_update"),
    ):
        result = repo.load_tool("tool")
        assert isinstance(result, Callable)


@patch("aic_core.agent.agent_hub.hf_hub_download")
@patch("aic_core.agent.agent_hub.importlib.util")
def test_load_structured_output(mock_importlib, mock_download):
    repo = AgentHub("test-repo")
    mock_download.return_value = "/path/to/model.py"

    # Setup mock module
    mock_module = Mock()
    mock_module.model = DummyModel
    mock_spec = Mock()
    mock_spec.loader = Mock()

    mock_importlib.spec_from_file_location.return_value = mock_spec
    mock_importlib.module_from_spec.return_value = mock_module

    with (
        patch("aic_core.agent.agent_hub.AgentHub._layzy_update"),
    ):
        result = repo.load_result_type("model")
        assert issubclass(result, BaseModel)


def test_upload_content():
    # Initialize the repo
    repo = AgentHub("test-repo")

    test_cases = [
        # (filename, content, subdir, expected_extension)
        ("test_tool", "def test_tool(): pass", "tools", ".py"),
        (
            "test_model",
            "from pydantic import BaseModel\nclass test_model(BaseModel): pass",
            "result_types",
            ".py",
        ),
        ("test_config", '{"key": "value"}', "agents", ".json"),
    ]

    for filename, content, subdir, extension in test_cases:
        with patch("aic_core.agent.agent_hub.upload_file") as mock_upload:
            # Call the method
            repo.upload_content(filename, content, subdir)

            # Verify upload_file was called with correct arguments
            mock_upload.assert_called_once_with(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=f"{subdir}/{filename}{extension}",
                repo_id=repo.repo_id,
                repo_type=repo.repo_type,
                commit_message=f"Update {filename}{extension}",
            )


def test_upload_content_invalid_subdir():
    repo = AgentHub("test-repo")

    with pytest.raises(ValueError, match="Invalid type: invalid_dir"):
        repo.upload_content("test", "content", "invalid_dir")


def test_upload_content_with_extension():
    repo = AgentHub("test-repo")

    with (
        patch("aic_core.agent.agent_hub.upload_file") as mock_upload,
    ):
        # Call with filename that already has extension
        repo.upload_content("test_tool.py", "content", "tools")

        # Verify correct handling of existing extension
        mock_upload.assert_called_once_with(
            path_or_fileobj=b"content",
            path_in_repo="tools/test_tool.py",
            repo_id=repo.repo_id,
            repo_type=repo.repo_type,
            commit_message="Update test_tool.py",
        )


def test_list_files_existing_directory():
    """Test listing files in an existing directory."""
    with (
        patch("aic_core.agent.agent_hub.snapshot_download") as mock_snapshot,
        patch("os.path.exists") as mock_exists,
        patch("os.path.isdir") as mock_isdir,
        patch("os.listdir") as mock_listdir,
        patch("os.path.isfile") as mock_isfile,
    ):
        # Setup mocks
        mock_snapshot.return_value = "/fake/repo/path"
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.py", "file2.py", "file3.json"]
        mock_isfile.return_value = True

        hub = AgentHub("test-repo")
        files = hub.list_files("tools")

        # Verify results
        assert files == ["file1", "file2", "file3"]
        mock_snapshot.assert_called_once_with(
            repo_id=hub.repo_id, repo_type=hub.repo_type, local_files_only=True
        )


def test_list_files_nonexistent_directory():
    """Test listing files in a non-existent directory."""
    with (
        patch("aic_core.agent.agent_hub.snapshot_download") as mock_snapshot,
        patch("os.path.exists") as mock_exists,
    ):
        mock_snapshot.return_value = "/fake/repo/path"
        mock_exists.return_value = False

        hub = AgentHub("test-repo")
        files = hub.list_files("nonexistent")

        assert files == []


def test_list_files_not_a_directory():
    """Test listing files when path exists but is not a directory."""
    with (
        patch("aic_core.agent.agent_hub.snapshot_download") as mock_snapshot,
        patch("os.path.exists") as mock_exists,
        patch("os.path.isdir") as mock_isdir,
    ):
        mock_snapshot.return_value = "/fake/repo/path"
        mock_exists.return_value = True
        mock_isdir.return_value = False

        hub = AgentHub("test-repo")
        files = hub.list_files("not_a_dir")

        assert files == []


@pytest.mark.parametrize("subdir", ["tools", "agents", "result_types"])
def test_delete_file_valid_subdirs(subdir):
    # Arrange
    repo_id = "test-repo"
    hub = AgentHub(repo_id)
    filename = "test_file"

    # Act
    with patch("aic_core.agent.agent_hub.delete_file") as mock_delete:
        hub.delete_file(filename, subdir)

    # Assert
    mock_delete.assert_called_once_with(
        path_in_repo=f"{subdir}/{filename}", repo_id=repo_id, repo_type="space"
    )


def test_delete_file_invalid_subdir():
    # Arrange
    repo_id = "test-repo"
    hub = AgentHub(repo_id)
    filename = "test_file"
    invalid_subdir = "invalid_dir"

    # Act & Assert
    with patch("aic_core.agent.agent_hub.delete_file") as mock_delete:
        hub.delete_file(filename, invalid_subdir)
        mock_delete.assert_called_once_with(
            path_in_repo=f"{invalid_subdir}/{filename}",
            repo_id=repo_id,
            repo_type="space",
        )


def test_layzy_update():
    """Test the _layzy_update method."""
    with (
        patch("aic_core.agent.agent_hub.snapshot_download") as mock_snapshot,
        patch("os.path.getmtime") as mock_getmtime,
        patch("os.utime") as mock_utime,
    ):
        # Setup
        hub = AgentHub("test-repo")
        mock_snapshot.return_value = "/fake/cache/path"

        # Test case 1: Cache is fresh (no update needed)
        mock_getmtime.return_value = time.time()  # Current time
        hub._layzy_update()
        # Should only call snapshot_download once with local_files_only=True
        assert mock_snapshot.call_count == 1
        mock_snapshot.assert_called_with(
            hub.repo_id, repo_type=hub.repo_type, local_files_only=True
        )
        mock_utime.assert_not_called()

        # Reset mocks
        mock_snapshot.reset_mock()
        mock_utime.reset_mock()

        # Test case 2: Cache is stale (update needed)
        mock_getmtime.return_value = time.time() - (
            hub.update_interval + 100
        )  # Old timestamp
        hub._layzy_update()
        # Should call snapshot_download twice:
        # 1. First with local_files_only=True
        # 2. Then without local_files_only to update
        assert mock_snapshot.call_count == 2
        mock_snapshot.assert_any_call(
            hub.repo_id, repo_type=hub.repo_type, local_files_only=True
        )
        mock_snapshot.assert_any_call(hub.repo_id, repo_type=hub.repo_type)
        mock_utime.assert_called_once_with("/fake/cache/path", None)


@patch("aic_core.agent.agent_hub.hf_hub_download")
def test_get_file_path_remote_download(mock_hf_download):
    """Test get_file_path when file is not found locally."""
    hub = AgentHub("test-repo")

    # Mock the _layzy_update method
    with patch.object(hub, "_layzy_update"):
        # Setup mock to first raise LocalEntryNotFoundError, then return a path
        mock_hf_download.side_effect = [
            LocalEntryNotFoundError("File not found locally"),
            "/path/to/downloaded/file.py",
        ]

        result = hub.get_file_path("test_tool", hub.tools_dir)

        # Verify the function was called twice with correct parameters
        assert mock_hf_download.call_count == 2

        # First call should try local files only
        mock_hf_download.assert_any_call(
            repo_id="test-repo",
            filename="test_tool.py",
            subfolder="tools",
            local_files_only=True,
            repo_type="space",
        )

        # Second call should try remote download
        mock_hf_download.assert_any_call(
            repo_id="test-repo",
            filename="test_tool.py",
            subfolder="tools",
            repo_type="space",
        )

        assert result == "/path/to/downloaded/file.py"
