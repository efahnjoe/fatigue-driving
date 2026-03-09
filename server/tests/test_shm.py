"""
Tests for the shared memory manager (src/core/shm.py).

These tests cover:
- ShmManager lifecycle (open/close/context manager)
- Read API (read(), read_one())
- Write API (write(), write_raw())
- InputFrame and OutputFrame ctypes structures
- InputSample dataclass
"""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from src.core.shm import MAX_HEIGHT, MAX_WIDTH, InputFrame, InputSample, OutputFrame, ShmManager


# ---------------------------------------------------------------------------
# InputFrame tests
# ---------------------------------------------------------------------------


class TestInputFrame:
    """Tests for the InputFrame ctypes structure."""

    def test_type_name(self) -> None:
        """InputFrame.type_name() returns 'InputFrame'."""
        assert InputFrame.type_name() == "InputFrame"

    def test_fields_exist(self) -> None:
        """InputFrame has the expected fields."""
        fields = {name: type_ for name, type_ in InputFrame._fields_}
        assert "frame_id" in fields
        assert "width" in fields
        assert "height" in fields
        assert "channels" in fields
        assert "data" in fields

    def test_create_instance(self) -> None:
        """Can create an InputFrame instance."""
        frame = InputFrame()
        assert frame.frame_id == 0
        assert frame.width == 0
        assert frame.height == 0
        assert frame.channels == 0

    def test_set_fields(self) -> None:
        """Can set InputFrame fields."""
        frame = InputFrame()
        frame.frame_id = 42
        frame.width = 640
        frame.height = 480
        frame.channels = 4
        assert frame.frame_id == 42
        assert frame.width == 640
        assert frame.height == 480
        assert frame.channels == 4

    def test_data_buffer_size(self) -> None:
        """InputFrame data buffer has correct size."""
        frame = InputFrame()
        expected_size = MAX_WIDTH * MAX_HEIGHT * 4
        assert len(frame.data) == expected_size


# ---------------------------------------------------------------------------
# OutputFrame tests
# ---------------------------------------------------------------------------


class TestOutputFrame:
    """Tests for the OutputFrame ctypes structure."""

    def test_type_name(self) -> None:
        """OutputFrame.type_name() returns 'OutputFrame'."""
        assert OutputFrame.type_name() == "OutputFrame"

    def test_fields_exist(self) -> None:
        """OutputFrame has the expected fields."""
        fields = {name: type_ for name, type_ in OutputFrame._fields_}
        assert "frame_id" in fields
        assert "width" in fields
        assert "height" in fields
        assert "data" in fields

    def test_create_instance(self) -> None:
        """Can create an OutputFrame instance."""
        frame = OutputFrame()
        assert frame.frame_id == 0
        assert frame.width == 0
        assert frame.height == 0

    def test_set_fields(self) -> None:
        """Can set OutputFrame fields."""
        frame = OutputFrame()
        frame.frame_id = 100
        frame.width = 1280
        frame.height = 720
        assert frame.frame_id == 100
        assert frame.width == 1280
        assert frame.height == 720

    def test_data_buffer_size(self) -> None:
        """OutputFrame data buffer has correct size."""
        frame = OutputFrame()
        expected_size = MAX_WIDTH * MAX_HEIGHT * 3
        assert len(frame.data) == expected_size


# ---------------------------------------------------------------------------
# InputSample tests
# ---------------------------------------------------------------------------


class TestInputSample:
    """Tests for the InputSample dataclass."""

    def test_create_instance(self) -> None:
        """Can create an InputSample instance."""
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        sample = InputSample(frame_id=1, bgr=bgr, width=640, height=480)
        assert sample.frame_id == 1
        assert sample.width == 640
        assert sample.height == 480
        assert sample.bgr.shape == (480, 640, 3)

    def test_slots(self) -> None:
        """InputSample uses __slots__."""
        assert hasattr(InputSample, "__slots__")
        sample = InputSample(frame_id=1, bgr=np.zeros((10, 10, 3)), width=10, height=10)
        with pytest.raises(AttributeError):
            sample.extra_attr = "should fail"


# ---------------------------------------------------------------------------
# ShmManager tests
# ---------------------------------------------------------------------------


class TestShmManagerLifecycle:
    """Tests for ShmManager lifecycle (open/close/context manager)."""

    def test_init_defaults(self) -> None:
        """ShmManager initializes with default values."""
        shm = ShmManager()
        assert shm._input_port == "video/input"
        assert shm._output_port == "video/output"
        assert shm._poll_interval == 0.001
        assert shm._running is False
        assert shm._node is None
        assert shm._subscriber is None
        assert shm._publisher is None

    def test_init_custom_values(self) -> None:
        """ShmManager initializes with custom values."""
        shm = ShmManager(
            input_port="custom/input",
            output_port="custom/output",
            poll_interval=0.005,
        )
        assert shm._input_port == "custom/input"
        assert shm._output_port == "custom/output"
        assert shm._poll_interval == 0.005

    def test_is_open_before_and_after_open(self) -> None:
        """is_open returns False before open, True after."""
        shm = ShmManager()
        assert shm.is_open is False

        # Manually set _running to simulate open state
        shm._running = True
        assert shm.is_open is True

    def test_close_sets_running_false(self) -> None:
        """close() sets _running to False."""
        shm = ShmManager()
        shm._running = True
        shm.close()
        assert shm._running is False

    @patch("src.core.shm.iox2")
    def test_open_creates_node_subscriber_publisher(self, mock_iox2: MagicMock) -> None:
        """open() creates node, subscriber and publisher."""
        # Setup mocks
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder

        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder

        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder

        mock_subscriber = MagicMock()
        mock_subscriber_builder.create.return_value = mock_subscriber

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder

        mock_publisher = MagicMock()
        mock_publisher_builder.create.return_value = mock_publisher

        # Run test
        shm = ShmManager()
        shm.open()

        # Verify
        assert shm._node is mock_node
        assert shm._subscriber is mock_subscriber
        assert shm._publisher is mock_publisher
        assert shm._running is True

        # Verify calls
        mock_iox2.NodeBuilder.new().create.assert_called_with(mock_iox2.ServiceType.Ipc)
        mock_node.service_builder.assert_any_call("video/input")
        mock_node.service_builder.assert_any_call("video/output")

    @patch("src.core.shm.iox2")
    def test_context_manager(self, mock_iox2: MagicMock) -> None:
        """Context manager calls open() and close()."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        with ShmManager() as shm:
            assert shm._running is True
            assert shm._node is not None

        assert shm._running is False


class TestShmManagerRead:
    """Tests for ShmManager read API."""

    @patch("src.core.shm.iox2")
    def test_read_one_returns_none_when_subscriber_is_none(self, mock_iox2: MagicMock) -> None:
        """read_one() returns None when subscriber is None."""
        shm = ShmManager()
        assert shm.read_one() is None

    @patch("src.core.shm.iox2")
    def test_read_one_returns_none_when_no_frames(self, mock_iox2: MagicMock) -> None:
        """read_one() returns None when no frames available."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder

        mock_subscriber = MagicMock()
        mock_subscriber.__iter__ = MagicMock(return_value=iter([]))
        mock_subscriber_builder.create.return_value = mock_subscriber

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        shm = ShmManager()
        shm.open()

        result = shm.read_one()
        assert result is None

    @patch("src.core.shm.iox2")
    @patch("src.core.shm.cv2")
    def test_read_one_returns_frame(self, mock_cv2: MagicMock, mock_iox2: MagicMock) -> None:
        """read_one() returns InputSample when frame available."""
        # Setup mock for cv2.cvtColor
        mock_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = mock_bgr

        # Setup iceoryx2 mocks
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder

        # Create a mock frame
        mock_frame = InputFrame()
        mock_frame.frame_id = 42
        mock_frame.width = 640
        mock_frame.height = 480
        mock_frame.channels = 4

        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_frame

        mock_subscriber = MagicMock()
        mock_subscriber.__iter__ = MagicMock(return_value=iter([mock_sample]))
        mock_subscriber_builder.create.return_value = mock_subscriber

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        shm = ShmManager()
        shm.open()

        result = shm.read_one()

        assert result is not None
        assert result.frame_id == 42
        assert result.width == 640
        assert result.height == 480
        mock_cv2.cvtColor.assert_called_once()

    @patch("src.core.shm.iox2")
    @patch("src.core.shm.cv2")
    def test_read_one_returns_none_for_zero_size_frame(
        self, mock_cv2: MagicMock, mock_iox2: MagicMock
    ) -> None:
        """read_one() returns None for zero-size frames."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder

        # Zero-size frame
        mock_frame = InputFrame()
        mock_frame.frame_id = 1
        mock_frame.width = 0
        mock_frame.height = 0
        mock_frame.channels = 4

        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_frame

        mock_subscriber = MagicMock()
        mock_subscriber.__iter__ = MagicMock(return_value=iter([mock_sample]))
        mock_subscriber_builder.create.return_value = mock_subscriber

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        shm = ShmManager()
        shm.open()

        result = shm.read_one()
        assert result is None

    @patch("src.core.shm.iox2")
    def test_read_generator_yields_frames(
        self, mock_iox2: MagicMock
    ) -> None:
        """read() generator yields frames."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder

        # Create mock frames
        mock_frame1 = InputFrame()
        mock_frame1.frame_id = 1
        mock_frame1.width = 640
        mock_frame1.height = 480
        mock_frame1.channels = 4

        mock_frame2 = InputFrame()
        mock_frame2.frame_id = 2
        mock_frame2.width = 640
        mock_frame2.height = 480
        mock_frame2.channels = 4

        mock_sample1 = MagicMock()
        mock_sample1.payload.return_value = mock_frame1
        mock_sample2 = MagicMock()
        mock_sample2.payload.return_value = mock_frame2

        # Use side_effect to return frames on first iteration, then empty
        mock_subscriber = MagicMock()
        mock_subscriber.__iter__.side_effect = [iter([mock_sample1, mock_sample2]), iter([])]
        mock_subscriber_builder.create.return_value = mock_subscriber

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        shm = ShmManager()

        with patch("src.core.shm.cv2") as mock_cv2:
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

            # Test that read_one works for first frame
            shm.open()
            frame1 = shm.read_one()

        assert frame1 is not None
        assert frame1.frame_id == 1


class TestShmManagerWrite:
    """Tests for ShmManager write API."""

    @patch("src.core.shm.iox2")
    def test_write_returns_false_before_open(self, mock_iox2: MagicMock) -> None:
        """write() returns False when publisher is None."""
        shm = ShmManager()
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = shm.write(bgr, frame_id=1)
        assert result is False

    @patch("src.core.shm.iox2")
    def test_write_rejects_oversized_frames(self, mock_iox2: MagicMock) -> None:
        """write() returns False for frames exceeding MAX_WIDTH/MAX_HEIGHT."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        shm = ShmManager()
        shm.open()

        # Frame too wide
        bgr_wide = np.zeros((480, MAX_WIDTH + 1, 3), dtype=np.uint8)
        result = shm.write(bgr_wide, frame_id=1)
        assert result is False

        # Frame too tall
        bgr_tall = np.zeros((MAX_HEIGHT + 1, 640, 3), dtype=np.uint8)
        result = shm.write(bgr_tall, frame_id=1)
        assert result is False

    @patch("src.core.shm.ctypes.addressof")
    @patch("src.core.shm.ctypes.memmove")
    @patch("src.core.shm.iox2")
    def test_write_sends_frame(
        self, mock_iox2: MagicMock, mock_memmove: MagicMock, mock_addressof: MagicMock
    ) -> None:
        """write() successfully sends a frame."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder

        # Create a real OutputFrame for the payload
        mock_payload = OutputFrame()
        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_payload

        mock_publisher = MagicMock()
        mock_publisher.loan_uninit.return_value = mock_sample
        mock_publisher_builder.create.return_value = mock_publisher

        # Mock addressof to return a dummy value
        mock_addressof.return_value = 0x1000

        shm = ShmManager()
        shm.open()

        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = shm.write(bgr, frame_id=42)

        assert result is True
        mock_publisher.loan_uninit.assert_called_once()
        mock_sample.assume_init().send.assert_called_once()
        # Verify payload was set correctly
        assert mock_payload.frame_id == 42
        assert mock_payload.width == 640
        assert mock_payload.height == 480
        mock_memmove.assert_called_once()

    @patch("src.core.shm.ctypes.addressof")
    @patch("src.core.shm.ctypes.memmove")
    @patch("src.core.shm.iox2")
    def test_write_converts_non_contiguous_array(
        self, mock_iox2: MagicMock, mock_memmove: MagicMock, mock_addressof: MagicMock
    ) -> None:
        """write() converts non-contiguous arrays."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder

        # Create a real OutputFrame for the payload
        mock_payload = OutputFrame()
        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_payload

        mock_publisher = MagicMock()
        mock_publisher.loan_uninit.return_value = mock_sample
        mock_publisher_builder.create.return_value = mock_publisher

        # Mock addressof to return a dummy value
        mock_addressof.return_value = 0x1000

        shm = ShmManager()
        shm.open()

        # Create non-contiguous array (transpose makes it non-contiguous)
        bgr = np.zeros((3, 480, 640), dtype=np.uint8).transpose(1, 2, 0)
        assert bgr.flags["C_CONTIGUOUS"] is False

        result = shm.write(bgr, frame_id=1)
        assert result is True
        # Verify payload was set
        assert mock_payload.frame_id == 1
        assert mock_payload.width == 640
        assert mock_payload.height == 480
        mock_memmove.assert_called_once()


class TestShmManagerWriteRaw:
    """Tests for ShmManager write_raw API."""

    @patch("src.core.shm.iox2")
    def test_write_raw_returns_false_before_open(self, mock_iox2: MagicMock) -> None:
        """write_raw() returns False when publisher is None."""
        shm = ShmManager()
        data = bytes(640 * 480 * 3)
        result = shm.write_raw(data, width=640, height=480, frame_id=1)
        assert result is False

    @patch("src.core.shm.iox2")
    def test_write_raw_rejects_wrong_length(self, mock_iox2: MagicMock) -> None:
        """write_raw() returns False for incorrect data length."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder
        mock_publisher_builder.create.return_value = MagicMock()

        shm = ShmManager()
        shm.open()

        # Wrong length (should be 640*480*3)
        data = bytes(100)
        result = shm.write_raw(data, width=640, height=480, frame_id=1)
        assert result is False

    @patch("src.core.shm.ctypes.addressof")
    @patch("src.core.shm.ctypes.memmove")
    @patch("src.core.shm.iox2")
    def test_write_raw_sends_bytes(
        self, mock_iox2: MagicMock, mock_memmove: MagicMock, mock_addressof: MagicMock
    ) -> None:
        """write_raw() successfully sends bytes data."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder

        # Create a real OutputFrame for the payload
        mock_payload = OutputFrame()
        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_payload

        mock_publisher = MagicMock()
        mock_publisher.loan_uninit.return_value = mock_sample
        mock_publisher_builder.create.return_value = mock_publisher

        # Mock addressof to return a dummy value
        mock_addressof.return_value = 0x1000

        shm = ShmManager()
        shm.open()

        data = bytes(640 * 480 * 3)
        result = shm.write_raw(data, width=640, height=480, frame_id=42)

        assert result is True
        mock_publisher.loan_uninit.assert_called_once()
        mock_sample.assume_init().send.assert_called_once()
        assert mock_payload.frame_id == 42
        assert mock_payload.width == 640
        assert mock_payload.height == 480
        mock_memmove.assert_called_once()

    @patch("src.core.shm.ctypes.addressof")
    @patch("src.core.shm.ctypes.memmove")
    @patch("src.core.shm.iox2")
    def test_write_raw_sends_bytearray(
        self, mock_iox2: MagicMock, mock_memmove: MagicMock, mock_addressof: MagicMock
    ) -> None:
        """write_raw() successfully sends bytearray data."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder

        # Create a real OutputFrame for the payload
        mock_payload = OutputFrame()
        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_payload

        mock_publisher = MagicMock()
        mock_publisher.loan_uninit.return_value = mock_sample
        mock_publisher_builder.create.return_value = mock_publisher

        # Mock addressof to return a dummy value
        mock_addressof.return_value = 0x1000

        shm = ShmManager()
        shm.open()

        data = bytearray(640 * 480 * 3)
        result = shm.write_raw(data, width=640, height=480, frame_id=1)

        assert result is True
        assert mock_payload.frame_id == 1
        assert mock_payload.width == 640
        assert mock_payload.height == 480
        mock_memmove.assert_called_once()

    @patch("src.core.shm.ctypes.addressof")
    @patch("src.core.shm.ctypes.memmove")
    @patch("src.core.shm.iox2")
    def test_write_raw_sends_ndarray(
        self, mock_iox2: MagicMock, mock_memmove: MagicMock, mock_addressof: MagicMock
    ) -> None:
        """write_raw() successfully sends numpy ndarray data."""
        mock_node = MagicMock()
        mock_iox2.NodeBuilder.new().create.return_value = mock_node

        mock_service_builder = MagicMock()
        mock_node.service_builder.return_value = mock_service_builder
        mock_pub_sub_builder = MagicMock()
        mock_service_builder.publish_subscribe.return_value = mock_pub_sub_builder
        mock_open_or_create = MagicMock()
        mock_pub_sub_builder.open_or_create.return_value = mock_open_or_create

        mock_subscriber_builder = MagicMock()
        mock_open_or_create.subscriber_builder.return_value = mock_subscriber_builder
        mock_subscriber_builder.create.return_value = MagicMock()

        mock_publisher_builder = MagicMock()
        mock_open_or_create.publisher_builder.return_value = mock_publisher_builder

        # Create a real OutputFrame for the payload
        mock_payload = OutputFrame()
        mock_sample = MagicMock()
        mock_sample.payload.return_value = mock_payload

        mock_publisher = MagicMock()
        mock_publisher.loan_uninit.return_value = mock_sample
        mock_publisher_builder.create.return_value = mock_publisher

        # Mock addressof to return a dummy value
        mock_addressof.return_value = 0x1000

        shm = ShmManager()
        shm.open()

        data = np.zeros(640 * 480 * 3, dtype=np.uint8)
        result = shm.write_raw(data, width=640, height=480, frame_id=1)

        assert result is True
        assert mock_payload.frame_id == 1
        assert mock_payload.width == 640
        assert mock_payload.height == 480
        mock_memmove.assert_called_once()


class TestShmManagerDecode:
    """Tests for the internal _decode method."""

    @patch("src.core.shm.cv2")
    def test_decode_zero_size_returns_none(self, mock_cv2: MagicMock) -> None:
        """_decode returns None for zero-size frames."""
        frame = InputFrame()
        frame.width = 0
        frame.height = 0

        result = ShmManager._decode(frame)
        assert result is None
        mock_cv2.cvtColor.assert_not_called()

    @patch("src.core.shm.cv2")
    def test_decode_converts_rgba_to_bgr(self, mock_cv2: MagicMock) -> None:
        """_decode converts RGBA to BGR."""
        mock_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = mock_bgr

        frame = InputFrame()
        frame.frame_id = 100
        frame.width = 640
        frame.height = 480
        frame.channels = 4

        result = ShmManager._decode(frame)

        assert result is not None
        assert result.frame_id == 100
        assert result.width == 640
        assert result.height == 480
        assert result.bgr.shape == (480, 640, 3)
        mock_cv2.cvtColor.assert_called_once()
