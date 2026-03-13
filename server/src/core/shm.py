"""
Iceoryx2 shared memory manager.

Responsibilities
----------------
- Owns the iceoryx2 node, subscriber and publisher lifecycle.
- Exposes a minimal read / write API; all frame-processing logic lives
  in the caller.

Typical usage::

    with ShmManager() as shm:
        for frame in shm.read():
            result = my_model(frame.bgr)
            shm.write(result, frame.frame_id)
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Generator

import cv2
import iceoryx2 as iox2
import numpy as np

logger = logging.getLogger(__name__)

MAX_WIDTH: int = 1920
MAX_HEIGHT: int = 1080


# ---------------------------------------------------------------------------
# Shared memory frame types — must match Rust #[repr(C)] structs exactly
# ---------------------------------------------------------------------------


class InputFrame(ctypes.Structure):
    """RGBA frame produced by the Rust frontend."""

    _fields_ = [
        ("frame_id", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("data", ctypes.c_uint8 * (MAX_WIDTH * MAX_HEIGHT * 4)),
    ]

    @staticmethod
    def type_name() -> str:
        return "InputFrame"


class OutputFrame(ctypes.Structure):
    """BGR frame consumed by the Rust frontend."""

    _fields_ = [
        ("frame_id", ctypes.c_uint64),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("data", ctypes.c_uint8 * (MAX_WIDTH * MAX_HEIGHT * 3)),
    ]

    @staticmethod
    def type_name() -> str:
        return "OutputFrame"


# ---------------------------------------------------------------------------
# Public data transfer object returned by ShmManager.read()
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InputSample:
    """
    A single frame received from shared memory.

    Attributes
    ----------
    frame_id:
        Monotonically increasing counter set by the Rust frontend.
        Pass it back to :meth:`ShmManager.write` for latency tracking.
    bgr:
        ``np.ndarray`` of shape ``(H, W, 3)``, ``dtype=uint8``, channel
        order **BGR** — ready for OpenCV or PyTorch (after colour conversion).
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    """

    frame_id: int
    bgr: np.ndarray
    width: int
    height: int


# ---------------------------------------------------------------------------
# Shared memory manager
# ---------------------------------------------------------------------------


class ShmManager:
    """
    Thin wrapper around iceoryx2 that owns the node / subscriber / publisher
    and exposes three methods:

    * :meth:`read`  — iterate incoming frames as :class:`InputSample`
    * :meth:`write` — publish a processed BGR frame back to the frontend
    * :meth:`write_raw` — publish a pre-built BGR ``ndarray`` with full control

    All frame processing belongs in the caller; this class does *not* run
    any processing loop.

    Parameters
    ----------
    input_port:
        Iceoryx2 service name the Rust frontend publishes to.
    output_port:
        Iceoryx2 service name the Rust frontend subscribes to.
    poll_interval:
        Seconds to sleep between subscriber polls when the queue is empty.
        Lower values reduce latency at the cost of CPU usage.
    """

    __slots__ = (
        "_input_port",
        "_output_port",
        "_poll_interval",
        "_running",
        "_node",
        "_subscriber",
        "_publisher",
    )

    def __init__(
        self,
        input_port: str = "video/input",
        output_port: str = "video/output",
        poll_interval: float = 0.001,
    ) -> None:
        self._input_port = input_port
        self._output_port = output_port
        self._poll_interval = poll_interval
        self._running = False

        self._node: iox2.Node | None = None
        self._subscriber: iox2.Subscriber[InputFrame] | None = None
        self._publisher: iox2.Publisher[OutputFrame] | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the iceoryx2 node, subscriber and publisher."""
        self._node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

        self._subscriber = (
            self._node.service_builder(iox2.ServiceName.new(self._input_port))
            .publish_subscribe(InputFrame)
            .open_or_create()
            .subscriber_builder()
            .create()
        )
        self._publisher = (
            self._node.service_builder(iox2.ServiceName.new(self._output_port))
            .publish_subscribe(OutputFrame)
            .open_or_create()
            .publisher_builder()
            .create()
        )

        self._running = True
        logger.info(
            "ShmManager opened  input=%s  output=%s",
            self._input_port,
            self._output_port,
        )

    def close(self) -> None:
        """Release all iceoryx2 resources."""
        self._running = False
        self._publisher = None
        self._subscriber = None
        self._node = None
        logger.info("ShmManager closed")

    @property
    def is_open(self) -> bool:
        return self._running

    # ── Read API ─────────────────────────────────────────────────────────────

    def read(self) -> Generator[InputSample, None, None]:
        """
        Iterate frames arriving from shared memory.

        Yields
        ------
        InputSample
            Each sample contains the decoded BGR image and metadata.
            The generator runs until :meth:`close` is called or the
            process receives ``KeyboardInterrupt``.

        Example
        -------
        ::

            with ShmManager() as shm:
                for sample in shm.read():
                    result = my_model(sample.bgr)
                    shm.write(result, sample.frame_id)
        """

        if not self._running:
            self.open()

        while self._running:
            if self._subscriber is None:
                break

            received = False
            while True:
                raw = self._subscriber.receive()
                if raw is None:
                    break

                sample = self._decode(raw.payload())
                if sample is not None:
                    received = True
                    yield sample

            if not received:
                time.sleep(self._poll_interval)

    def read_one(self) -> InputSample | None:
        """
        Return the next available frame without blocking, or ``None``.

        Useful when the caller manages its own event loop.

        Example
        -------
        ::

            shm.open()
            while True:
                sample = shm.read_one()
                if sample:
                    shm.write(my_model(sample.bgr), sample.frame_id)
                else:
                    time.sleep(0.001)
        """

        if self._subscriber is None:
            return None

        raw = self._subscriber.receive()
        if raw is not None:
            return self._decode(raw.payload())
        return None

    # ── Write API ─────────────────────────────────────────────────────────────

    def write(self, bgr: np.ndarray, frame_id: int = 0) -> bool:
        """
        Publish a processed BGR frame to the ``video/output`` service.

        Parameters
        ----------
        bgr:
            Processed image, shape ``(H, W, 3)``, ``dtype=uint8``, BGR order.
            Resolution must not exceed ``MAX_WIDTH × MAX_HEIGHT``.
        frame_id:
            Echo the ``frame_id`` from the originating :class:`InputSample`
            so the Rust frontend can measure round-trip latency.

        Returns
        -------
        bool
            ``True`` if the frame was sent successfully.

        Example
        -------
        ::

            sample = shm.read_one()
            if sample:
                result = my_model(sample.bgr)   # (H, W, 3) uint8 BGR
                shm.write(result, sample.frame_id)
        """

        if self._publisher is None:
            logger.warning("write() called before open()")
            return False

        h, w = bgr.shape[:2]
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            logger.error(
                "Frame %dx%d exceeds maximum %dx%d; skipping",
                w,
                h,
                MAX_WIDTH,
                MAX_HEIGHT,
            )
            return False

        # Ensure contiguous uint8 BGR before memmove
        if not bgr.flags["C_CONTIGUOUS"] or bgr.dtype != np.uint8:
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

        out = self._publisher.loan_uninit()
        payload = out.payload().contents
        payload.frame_id = frame_id
        payload.width = w
        payload.height = h
        ctypes.memmove(
            ctypes.addressof(payload.data),
            bgr.ctypes.data,
            w * h * 3,
        )
        out.assume_init().send()
        return True

    def write_raw(
        self,
        data: bytes | bytearray | np.ndarray,
        width: int,
        height: int,
        frame_id: int = 0,
    ) -> bool:
        """
        Publish a raw BGR byte buffer without any numpy conversion.

        Use this when you already have a flat ``bytes``/``bytearray``
        or a 1-D ``ndarray`` and want to skip the reshape overhead.

        Parameters
        ----------
        data:
            Flat BGR bytes of length ``width * height * 3``.
        width, height:
            Frame dimensions in pixels.
        frame_id:
            Passed through to the Rust frontend unchanged.

        Returns
        -------
        bool
            ``True`` if the frame was sent successfully.
        """

        if self._publisher is None:
            logger.warning("write_raw() called before open()")
            return False

        expected = width * height * 3
        actual = len(data)
        if actual != expected:
            logger.error("write_raw: data length %d != expected %d", actual, expected)
            return False

        out = self._publisher.loan_uninit()
        payload = out.payload().contents
        payload.frame_id = frame_id
        payload.width = width
        payload.height = height

        src_ptr = (
            data.ctypes.data
            if isinstance(data, np.ndarray)
            else ctypes.cast(
                (ctypes.c_char * len(data)).from_buffer_copy(data),
                ctypes.c_void_p,
            ).value
        )
        ctypes.memmove(ctypes.addressof(payload.data), src_ptr, expected)
        out.assume_init().send()
        return True

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> ShmManager:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _decode(frame_ptr: InputFrame) -> InputSample | None:
        """Convert a raw InputFrame into an InputSample (RGBA → BGR)."""

        frame = frame_ptr.contents
        w, h = frame.width, frame.height
        if w == 0 or h == 0:
            return None

        # Zero-copy view into shared memory
        rgba = np.frombuffer(
            (ctypes.c_uint8 * (w * h * 4)).from_address(ctypes.addressof(frame.data)),
            dtype=np.uint8,
        ).reshape(h, w, 4)

        return InputSample(
            frame_id=int(frame.frame_id),
            bgr=cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR),
            width=w,
            height=h,
        )
