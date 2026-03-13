"""
Tests for the model manager (src/models/manager.py).

These tests cover:
- ModelManager singleton pattern
- Model loading (load())
- Model inference (run())
- Cache management (clear_cache())
- Helper functions (load_model(), run_model())
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.manager import (
    DEFAULT_PROVIDERS,
    ModelManager,
    load_model,
    run_model,
)


# ---------------------------------------------------------------------------
# Module-level constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_default_providers_not_empty(self) -> None:
        """DEFAULT_PROVIDERS should not be empty."""
        assert len(DEFAULT_PROVIDERS) > 0

    def test_default_providers_contains_cpu(self) -> None:
        """DEFAULT_PROVIDERS should always contain CPUExecutionProvider."""
        assert "CPUExecutionProvider" in DEFAULT_PROVIDERS


# ---------------------------------------------------------------------------
# ModelManager singleton tests
# ---------------------------------------------------------------------------


class TestModelManagerSingleton:
    """Tests for ModelManager singleton pattern."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelManager._instance = None

    def test_singleton_returns_same_instance(self) -> None:
        """Multiple calls to ModelManager() return the same instance."""
        manager1 = ModelManager()
        manager2 = ModelManager()
        assert manager1 is manager2

    def test_new_returns_instance(self) -> None:
        """ModelManager() returns a ModelManager instance."""
        manager = ModelManager()
        assert isinstance(manager, ModelManager)

    def test_sessions_initialized(self) -> None:
        """ModelManager initializes _sessions dict."""
        manager = ModelManager()
        assert hasattr(manager, "_sessions")
        assert isinstance(manager._sessions, dict)


# ---------------------------------------------------------------------------
# ModelManager.load() tests
# ---------------------------------------------------------------------------


class TestModelManagerLoad:
    """Tests for ModelManager.load() method."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelManager._instance = None

    @patch("src.models.manager.ort.InferenceSession")
    def test_load_existing_session_returns_cached(
        self, mock_session: MagicMock
    ) -> None:
        """load() returns cached session if model already loaded."""
        manager = ModelManager()
        mock_session_instance = MagicMock()
        manager._sessions["test_model.onnx"] = mock_session_instance

        result = manager.load("test_model.onnx")

        assert result is mock_session_instance
        mock_session.assert_not_called()

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_load_creates_session(
        self, mock_session: MagicMock
    ) -> None:
        """load() creates new InferenceSession for unloaded model."""
        # Setup mock
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        # Create mock model path with exists() returning True
        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()
            result = manager.load("test_model.onnx")

            assert result is mock_session_instance
            mock_session.assert_called_once()

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_load_uses_default_providers(
        self, mock_session: MagicMock
    ) -> None:
        """load() uses DEFAULT_PROVIDERS when no providers specified."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()
            manager.load("test_model.onnx")

            call_args = mock_session.call_args
            assert call_args.kwargs["providers"] == DEFAULT_PROVIDERS

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_load_uses_custom_providers(
        self, mock_session: MagicMock
    ) -> None:
        """load() uses custom providers when specified."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        custom_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()
            manager.load("test_model.onnx", providers=custom_providers)

            call_args = mock_session.call_args
            assert call_args.kwargs["providers"] == custom_providers

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    def test_load_raises_file_not_found(self) -> None:
        """load() raises FileNotFoundError when model file doesn't exist."""
        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = False
            mock_model_path.__str__.return_value = "/fake/models/missing_model.onnx"
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()

            with pytest.raises(FileNotFoundError) as exc_info:
                manager.load("missing_model.onnx")

            assert "Cannot find model in:" in str(exc_info.value)
            assert "missing_model.onnx" in str(exc_info.value)

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_load_caches_session(
        self, mock_session: MagicMock
    ) -> None:
        """load() caches session for subsequent calls."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()

            # First call creates session
            result1 = manager.load("test_model.onnx")
            # Second call returns cached session
            result2 = manager.load("test_model.onnx")

            assert result1 is result2
            assert "test_model.onnx" in manager._sessions
            mock_session.assert_called_once()


# ---------------------------------------------------------------------------
# ModelManager.run() tests
# ---------------------------------------------------------------------------


class TestModelManagerRun:
    """Tests for ModelManager.run() method."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelManager._instance = None

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_run_executes_model(self, mock_session_cls: MagicMock) -> None:
        """run() executes model inference."""
        # Setup mock session
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        # Mock get_inputs to return input name
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        # Mock run to return output
        mock_output = [[1, 2, 3]]
        mock_session.run.return_value = mock_output

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()
            input_data = [[0.1, 0.2, 0.3]]
            result = manager.run("test_model.onnx", input_data)

            assert result == mock_output
            mock_session.run.assert_called_once_with(
                None, {"input": input_data}
            )

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_run_loads_model_if_not_loaded(
        self, mock_session_cls: MagicMock
    ) -> None:
        """run() loads model if not already loaded."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [[1, 2, 3]]

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()
            manager.run("test_model.onnx", [[0.1, 0.2, 0.3]])

            mock_session_cls.assert_called_once()

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_run_uses_first_input(self, mock_session_cls: MagicMock) -> None:
        """run() uses the first input from get_inputs()."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        # Mock multiple inputs, should use first one
        mock_input1 = MagicMock()
        mock_input1.name = "input1"
        mock_input2 = MagicMock()
        mock_input2.name = "input2"
        mock_session.get_inputs.return_value = [mock_input1, mock_input2]
        mock_session.run.return_value = [[1, 2, 3]]

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            manager = ModelManager()
            manager.run("test_model.onnx", [[0.1, 0.2, 0.3]])

            call_args = mock_session.run.call_args
            assert call_args[0][1]["input1"] == [[0.1, 0.2, 0.3]]


# ---------------------------------------------------------------------------
# ModelManager.clear_cache() tests
# ---------------------------------------------------------------------------


class TestModelManagerClearCache:
    """Tests for ModelManager.clear_cache() method."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelManager._instance = None

    def test_clear_cache_removes_all_sessions(self) -> None:
        """clear_cache() removes all cached sessions."""
        manager = ModelManager()
        manager._sessions["model1.onnx"] = MagicMock()
        manager._sessions["model2.onnx"] = MagicMock()

        manager.clear_cache()

        assert len(manager._sessions) == 0

    def test_clear_cache_on_empty(self) -> None:
        """clear_cache() works on empty cache."""
        manager = ModelManager()
        initial_len = len(manager._sessions)

        manager.clear_cache()

        assert len(manager._sessions) == initial_len


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for load_model() and run_model() helper functions."""

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        ModelManager._instance = None

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_load_model_calls_manager(
        self, mock_session: MagicMock
    ) -> None:
        """load_model() delegates to ModelManager.load()."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            result = load_model("test_model.onnx")

            assert result is mock_session_instance

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_load_model_with_custom_providers(
        self, mock_session: MagicMock
    ) -> None:
        """load_model() passes custom providers to ModelManager."""
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        custom_providers = ["CUDAExecutionProvider"]

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            result = load_model("test_model.onnx", providers=custom_providers)

            call_args = mock_session.call_args
            assert call_args.kwargs["providers"] == custom_providers

    @patch("src.models.manager.MODELS_DIR", Path("/fake/models"))
    @patch("src.models.manager.ort.InferenceSession")
    def test_run_model_calls_manager(self, mock_session_cls: MagicMock) -> None:
        """run_model() delegates to ModelManager.run()."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [[1, 2, 3]]

        with patch("src.models.manager.MODELS_DIR") as mock_models_dir:
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_models_dir.__truediv__.return_value = mock_model_path

            result = run_model("test_model.onnx", [[0.1, 0.2, 0.3]])

            assert result == [[1, 2, 3]]
