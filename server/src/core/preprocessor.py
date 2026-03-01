"""Video source and frame preprocessing utilities."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Literal

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoSize:
    """Video frame size."""

    width: int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def __iter__(self) -> Generator[int, None, None]:
        yield self.width
        yield self.height

    def to_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)


# ---------------------------------------------------------------------------
# WebSocket Adapters
# ---------------------------------------------------------------------------


class WebSocketAdapter(ABC):
    """Abstract adapter for different WebSocket implementations."""

    @abstractmethod
    async def receive_bytes(self) -> bytes:
        """Receive raw bytes from the WebSocket connection."""
        ...

    @abstractmethod
    async def send_bytes(self, data: bytes) -> None:
        ...


class FastAPIWebSocketAdapter(WebSocketAdapter):
    """Adapter for FastAPI's WebSocket (starlette)."""

    __slots__ = ("_ws",)

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def receive_bytes(self) -> bytes:
        return await self._ws.receive_bytes()

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)


class WebsocketsLibAdapter(WebSocketAdapter):
    """Adapter for the `websockets` library connection object."""

    __slots__ = ("_ws",)

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def receive_bytes(self) -> bytes:
        data = await self._ws.recv()
        if isinstance(data, str):
            raise ValueError("Expected binary frame, got text frame")
        return data

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send(data)


def _make_adapter(ws: Any) -> WebSocketAdapter:
    """Auto-detect WebSocket implementation and return the correct adapter."""
    if isinstance(ws, WebSocketAdapter):
        return ws

    cls_name = type(ws).__name__
    module = getattr(type(ws), "__module__", "")

    if "starlette" in module or cls_name == "WebSocket":
        return FastAPIWebSocketAdapter(ws)

    if hasattr(ws, "recv"):
        return WebsocketsLibAdapter(ws)

    raise TypeError(
        f"Unsupported WebSocket type: {type(ws)}. "
        "Pass a FastAPIWebSocketAdapter or WebsocketsLibAdapter explicitly."
    )


# ---------------------------------------------------------------------------
# Base Video Source
# ---------------------------------------------------------------------------


class BaseVideoSource(ABC):
    """Abstract base class for video sources."""

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @property
    @abstractmethod
    def size(self) -> VideoSize:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class Cv2VideoSourceMixin:
    """Mixin for OpenCV-based video sources."""

    cap: cv2.VideoCapture

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def size(self) -> VideoSize:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return VideoSize(width=w, height=h)

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


# ---------------------------------------------------------------------------
# Video Sources
# ---------------------------------------------------------------------------


class LocalFileVideoSource(Cv2VideoSourceMixin, BaseVideoSource):
    """Video source from local file."""

    __slots__ = ("path", "cap")

    def __init__(self, filepath: str | Path) -> None:
        self.path = Path(filepath)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {filepath}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {filepath}")

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield frames as BGR numpy arrays."""
        while (ret := self.cap.read()[0]):
            yield self.cap.read()[1]

    def __enter__(self) -> LocalFileVideoSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class WebSocketVideoSource(BaseVideoSource):
    """
    Video source from an already-connected WebSocket object.

    Supports:
    - FastAPI / Starlette WebSocket
    - websockets library connection object
    - Custom WebSocketAdapter subclasses
    """

    __slots__ = ("_adapter", "_frame_queue", "_running")

    def __init__(self, ws: Any, max_queue_size: int = 30) -> None:
        self._adapter = _make_adapter(ws)
        self._frame_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._running = False

    @property
    def fps(self) -> float:
        return 0.0

    @property
    def size(self) -> VideoSize:
        return VideoSize(width=0, height=0)

    async def _receive_loop(self) -> None:
        self._running = True
        try:
            while self._running:
                try:
                    data = await self._adapter.receive_bytes()
                except Exception:
                    break

                frame = _decode_frame(data)
                if frame is None:
                    continue

                try:
                    self._frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    try:
                        self._frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await self._frame_queue.put(frame)
        finally:
            self._running = False
            await self._frame_queue.put(None)

    @staticmethod
    def decode_frame(data: bytes) -> np.ndarray | None:
        """Decode a single raw bytes frame."""
        return _decode_frame(data)

    async def frames(self) -> AsyncGenerator[np.ndarray, None]:
        """Async generator yielding frames."""
        receive_task = asyncio.create_task(self._receive_loop())
        try:
            while True:
                frame = await self._frame_queue.get()
                if frame is None:
                    break
                yield frame
        finally:
            self.stop()
            receive_task.cancel()

    def stop(self) -> None:
        self._running = False

    def release(self) -> None:
        self.stop()

    async def __aenter__(self) -> WebSocketVideoSource:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class CameraVideoSource(Cv2VideoSourceMixin, BaseVideoSource):
    """Video source from camera device."""

    __slots__ = ("device_id", "cap")

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera device: {device_id}")

    def frames(self) -> Generator[np.ndarray, None, None]:
        while (ret := self.cap.read()[0]):
            yield self.cap.read()[1]

    def __enter__(self) -> CameraVideoSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


VideoSourceType = LocalFileVideoSource | WebSocketVideoSource | CameraVideoSource


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


class SourceType(Enum):
    LOCAL = "local"
    WEBSOCKET = "websocket"
    CAMERA = "camera"


def create_video_source(
    source: str | int | Any,
    source_type: SourceType | None = None,
) -> VideoSourceType:
    """
    Create video source from path, device ID, or connected WebSocket object.

    Args:
        source: File path / device ID / connected WebSocket object
        source_type: Force specific type, auto-detect if None

    Returns:
        Corresponding video source instance
    """
    if source_type is None:
        if isinstance(source, int):
            source_type = SourceType.CAMERA
        elif isinstance(source, str):
            source_type = SourceType.LOCAL
        else:
            source_type = SourceType.WEBSOCKET

    match source_type:
        case SourceType.LOCAL:
            return LocalFileVideoSource(source)
        case SourceType.WEBSOCKET:
            return WebSocketVideoSource(source)
        case SourceType.CAMERA:
            return CameraVideoSource(source if isinstance(source, int) else 0)
        case _:
            raise ValueError(f"Unknown source type: {source_type}")


# ---------------------------------------------------------------------------
# Internal Utilities
# ---------------------------------------------------------------------------


def _decode_frame(data: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes to BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Frame Preprocessing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Frame preprocessing configuration."""

    size: tuple[int, int] | None = (640, 480)
    to_gray: bool = False
    normalize: bool = False
    color_order: Literal["BGR", "RGB"] = "BGR"


def preprocess_frame(
    frame: np.ndarray,
    size: tuple[int, int] | None = (640, 480),
    to_gray: bool = False,
    normalize: bool = False,
    color_order: Literal["BGR", "RGB"] = "BGR",
) -> np.ndarray:
    """Preprocess a video frame."""
    if size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    if to_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if color_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif color_order == "RGB":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if normalize:
        frame = frame.astype(np.float32) / 255.0

    return frame


def normalize_frame(
    frame: np.ndarray, mean: list[float], std: list[float]
) -> np.ndarray:
    """Normalize frame with given mean and std."""
    frame = frame.astype(np.float32)
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    return (frame / 255.0 - mean) / std


def transpose_frame(
    frame: np.ndarray, format: Literal["HWC", "CHW"] = "CHW"
) -> np.ndarray:
    """Transpose frame between HWC and CHW formats."""
    return np.transpose(frame, (2, 0, 1)) if format == "CHW" else frame
