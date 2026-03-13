//! # Shared Memory Manager (Rust side)
//!
//! Zero-copy video frame pipeline built on [iceoryx2](https://github.com/eclipse-iceoryx/iceoryx2),
//! mirroring the naming and interface conventions of the Python-side `ShmManager`.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────┐  InputSample   ┌─────────────┐  InputFrame (SHM)     ┌──────────────┐
//!  │    Caller    │ ─────────────► │  ShmWriter  │ ────────────────────► │  Python side │
//!  │ (Tokio async)│                │             │   input_port          │  ShmManager  │
//!  └──────────────┘                └─────────────┘                       └──────┬───────┘
//!                                                                               │ OutputFrame (SHM)
//!  ┌──────────────┐  OutputSample  ┌─────────────┐                              │  output_port
//!  │    Caller    │ ◄───────────── │  ShmReader  │ ◄────────────────────────────┘
//!  │ (Tokio async)│                │             │
//!  └──────────────┘                └─────────────┘
//! ```
//!
//! | Direction | SHM type | Default service name | Python counterpart |
//! |-----------|----------|---------------------|--------------------|
//! | Rust → Python | [`InputFrame`]  | `video/input`  | `InputSample` (Python reads)  |
//! | Python → Rust | [`OutputFrame`] | `video/output` | `write(bgr, frame_id)` |
//!
//! ## Quick start
//!
//! ### Unified manager
//!
//! ```rust,no_run
//! #[tokio::main]
//! async fn main() {
//!     let mut shm = ShmManager::new(ShmConfig::default());
//!     shm.open();
//!
//!     // Send a frame
//!     shm.writer().write_raw(0, 1280, 720, vec![0u8; 1280 * 720 * 4]).await.unwrap();
//!
//!     // Await the processed result
//!     if let Some(sample) = shm.reader().read().await {
//!         println!("frame {}: {}×{}", sample.frame_id, sample.width, sample.height);
//!         let _url = sample.to_data_url();
//!     }
//!
//!     shm.close();
//! }
//! ```
//!
//! ### Writer / reader handles used independently
//!
//! ```rust,no_run
//! let writer = ShmWriter::open(ShmConfig::default());
//! writer.write_raw(0, 640, 480, vec![128u8; 640 * 480 * 4]).await.unwrap();
//!
//! let mut reader = ShmReader::open(ShmConfig::default());
//! let sample = reader.read_one(); // non-blocking
//! ```

use base64::{Engine, engine::general_purpose};
use core::ptr::addr_of_mut;
use iceoryx2::prelude::*;
use image::{ImageFormat, RgbaImage};
use std::io::Cursor;
use std::time::Duration;
use tokio::sync::mpsc;

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────

/// Maximum supported frame width in pixels; matches Python-side `MAX_WIDTH`.
pub const MAX_WIDTH: usize = 1920;

/// Maximum supported frame height in pixels; matches Python-side `MAX_HEIGHT`.
pub const MAX_HEIGHT: usize = 1080;

// ─────────────────────────────────────────────
// Shared-memory frame types
// (#[repr(C)] — layout must match Python ctypes structs exactly)
// ─────────────────────────────────────────────

/// Input frame written by Rust and read by Python as `InputFrame(ctypes.Structure)`.
///
/// Field order matches the Python definition exactly:
/// ```python
/// _fields_ = [
///     ("frame_id",  c_uint64),
///     ("timestamp", c_uint64),
///     ("width",     c_uint32),
///     ("height",    c_uint32),
///     ("channels",  c_uint32),
///     ("data",      c_uint8 * (MAX_WIDTH * MAX_HEIGHT * 4)),
/// ]
/// ```
///
/// # Note
///
/// The struct is approximately 8 MB. **Do not** allocate it on the stack;
/// always construct it in-place via `loan_uninit()`.
#[repr(C)]
#[derive(Debug, Clone, Copy, ZeroCopySend)]
#[type_name("InputFrame")]
pub struct InputFrame {
    /// Monotonically increasing frame counter maintained by [`ShmWriter`].
    pub frame_id: u64,
    /// Unix timestamp (microseconds) recorded at publish time; lets Python
    /// measure receive latency.
    pub timestamp: u64,
    /// Actual image width in pixels; must be ≤ [`MAX_WIDTH`].
    pub width: u32,
    /// Actual image height in pixels; must be ≤ [`MAX_HEIGHT`].
    pub height: u32,
    /// Channels per pixel; always `4` (RGBA).
    pub channels: u32,
    /// Row-major RGBA pixel data; valid range is `data[..width * height * 4]`.
    pub data: [u8; MAX_WIDTH * MAX_HEIGHT * 4],
}

/// Output frame written by Python and read by Rust via [`ShmReader`].
///
/// Corresponds to the Python-side definition:
/// ```python
/// _fields_ = [
///     ("frame_id", c_uint64),
///     ("width",    c_uint32),
///     ("height",   c_uint32),
///     ("data",     c_uint8 * (MAX_WIDTH * MAX_HEIGHT * 3)),
/// ]
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, ZeroCopySend)]
#[type_name("OutputFrame")]
pub struct OutputFrame {
    /// Frame counter echoed from the originating [`InputFrame`]; used to
    /// measure round-trip latency.
    pub frame_id: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major BGR pixel data (OpenCV default); valid range is
    /// `data[..width * height * 3]`.
    pub data: [u8; MAX_WIDTH * MAX_HEIGHT * 3],
}

impl PlacementDefault for InputFrame {
    /// Zero-initialises an [`InputFrame`] in-place inside shared memory.
    ///
    /// # Safety
    ///
    /// `ptr` must point to at least `size_of::<InputFrame>()` bytes of valid,
    /// writable memory. The memory need not be initialised before the call.
    /// Every field is written via `addr_of_mut!`, so no reference to
    /// uninitialised memory is ever created.
    unsafe fn placement_default(ptr: *mut Self) {
        unsafe {
            addr_of_mut!((*ptr).frame_id).write(0);
            addr_of_mut!((*ptr).timestamp).write(0);
            addr_of_mut!((*ptr).width).write(0);
            addr_of_mut!((*ptr).height).write(0);
            addr_of_mut!((*ptr).channels).write(4);
            core::ptr::write_bytes(
                addr_of_mut!((*ptr).data) as *mut u8,
                0,
                MAX_WIDTH * MAX_HEIGHT * 4,
            );
        }
    }
}

impl PlacementDefault for OutputFrame {
    /// Zero-initialises an [`OutputFrame`] in-place inside shared memory.
    ///
    /// # Safety
    ///
    /// Same requirements as [`InputFrame::placement_default`].
    unsafe fn placement_default(ptr: *mut Self) {
        unsafe {
            addr_of_mut!((*ptr).frame_id).write(0);
            addr_of_mut!((*ptr).width).write(0);
            addr_of_mut!((*ptr).height).write(0);
            core::ptr::write_bytes(
                addr_of_mut!((*ptr).data) as *mut u8,
                0,
                MAX_WIDTH * MAX_HEIGHT * 3,
            );
        }
    }
}

// ─────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────

/// Errors that can be produced by the shared-memory pipeline.
///
/// Variants are aligned with the log messages emitted by the Python-side
/// `ShmManager` so that both ends report consistent diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum ShmError {
    /// iceoryx2 node initialisation failed.
    #[error("shm node creation failed: {0}")]
    NodeInit(String),

    /// iceoryx2 service creation or open failed.
    #[error("shm service creation failed: {0}")]
    ServiceInit(String),

    /// The internal Tokio channel is closed, meaning the background thread
    /// has exited.
    #[error("shm channel closed")]
    ChannelClosed,

    /// The supplied pixel buffer length does not equal `width × height × channels`.
    ///
    /// Mirrors Python: `logger.error("write_raw: data length %d != expected %d", ...)`.
    #[error("data length {actual} != expected {expected} (frame {width}×{height} ch={channels})")]
    DataLengthMismatch {
        actual: usize,
        expected: usize,
        width: u32,
        height: u32,
        channels: u32,
    },

    /// Frame dimensions exceed [`MAX_WIDTH`] × [`MAX_HEIGHT`].
    ///
    /// Mirrors Python: `logger.error("Frame %dx%d exceeds maximum %dx%d; skipping", ...)`.
    #[error("frame {width}×{height} exceeds maximum {MAX_WIDTH}×{MAX_HEIGHT}")]
    FrameExceedsMaximum { width: u32, height: u32 },
}

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Construction parameters for [`ShmWriter`], [`ShmReader`], and [`ShmManager`].
///
/// Field names mirror the `__init__` parameters of the Python `ShmManager`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShmConfig {
    /// iceoryx2 service name for the write path (Rust publishes, Python
    /// subscribes). Defaults to `"video/input"`.
    pub input_port: String,
    /// iceoryx2 service name for the read path (Python publishes, Rust
    /// subscribes). Defaults to `"video/output"`.
    pub output_port: String,
    /// Subscriber polling interval. Lower values reduce latency at the cost
    /// of CPU usage. Defaults to 1 ms.
    pub poll_interval: Duration,
    /// Internal Tokio channel capacity. When full, [`ShmWriter::write`] will
    /// back-pressure the caller. Defaults to 4.
    pub channel_capacity: usize,
}

impl Default for ShmConfig {
    fn default() -> Self {
        Self {
            input_port: "video/input".into(),
            output_port: "video/output".into(),
            poll_interval: Duration::from_millis(1),
            channel_capacity: 4,
        }
    }
}

// ─────────────────────────────────────────────
// InputSample — frame submitted to ShmWriter
// ─────────────────────────────────────────────

/// An RGBA frame submitted by the caller to [`ShmWriter`].
///
/// Corresponds to the `InputSample` that the Python side receives from
/// `ShmManager.read()`.
///
/// # Example
///
/// ```rust
/// let sample = InputSample {
///     frame_id: 0,
///     width: 1280,
///     height: 720,
///     rgba: vec![0u8; 1280 * 720 * 4],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct InputSample {
    /// Caller-managed frame counter; written as-is to
    /// [`InputFrame::frame_id`].
    pub frame_id: u64,
    /// Image width in pixels; must be ≤ [`MAX_WIDTH`].
    pub width: u32,
    /// Image height in pixels; must be ≤ [`MAX_HEIGHT`].
    pub height: u32,
    /// RGBA pixel data; length must equal exactly `width * height * 4`.
    pub rgba: Vec<u8>,
}

// ─────────────────────────────────────────────
// OutputSample — frame delivered by ShmReader
// ─────────────────────────────────────────────

/// A processed frame read from shared memory and delivered by [`ShmReader`].
///
/// The BGR→RGBA conversion is performed internally; the alpha channel is
/// always 255. Corresponds to the data written by the Python-side
/// `ShmManager.write(bgr, frame_id)`.
#[derive(Debug, Clone)]
pub struct OutputSample {
    /// Frame counter echoed from the originating [`InputFrame`]; can be used
    /// to measure end-to-end latency.
    pub frame_id: u64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// RGBA pixel data; length is `width * height * 4`; alpha is always 255.
    pub rgba: Vec<u8>,

    /// Cached PNG data URL suitable for use in `<img src="...">`.
    pub cached_data_url: Option<String>,
}

impl OutputSample {
    /// Encodes the frame as a JPEG data URL suitable for use in
    /// `<img src="...">`.
    ///
    /// > **Performance note**: Base64 encoding is performed on every call.
    /// > Cache the result if the same frame is rendered multiple times.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # let sample: OutputSample = todo!();
    /// let src = sample.to_data_url();
    /// // → "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    /// ```
    pub fn to_data_url(&self) -> String {
        let img_rgba = RgbaImage::from_raw(self.width, self.height, self.rgba.clone())
            .expect("RGBA length mismatch with dimensions");

        let img_rgb = image::DynamicImage::ImageRgba8(img_rgba).into_rgb8();

        let mut buffer = Cursor::new(Vec::new());
        img_rgb
            .write_to(&mut buffer, ImageFormat::Jpeg)
            .expect("encoding failed");

        let b64 = general_purpose::STANDARD.encode(buffer.into_inner());
        format!("data:image/jpeg;base64,{}", b64)
    }
}

// ─────────────────────────────────────────────
// ShmWriter — write side
// ─────────────────────────────────────────────

/// Async handle for publishing RGBA video frames to iceoryx2 shared memory.
///
/// Mirrors the Python-side `ShmManager.write` / `ShmManager.write_raw` API.
///
/// The handle is [`Clone`]; all clones share the same background publisher
/// thread.
///
/// # Back-pressure
///
/// When the internal channel is full, [`ShmWriter::write`] will await rather
/// than drop frames. Use `tokio::select!` with a timeout to implement
/// drop-on-overflow semantics if required.
#[derive(Clone)]
pub struct ShmWriter {
    tx: mpsc::Sender<InputSample>,
}

impl ShmWriter {
    /// Spawns the background publisher thread and returns a writer handle.
    ///
    /// Mirrors the publisher setup inside `ShmManager.open()` on the Python
    /// side. Returns immediately; iceoryx2 initialisation happens on the
    /// background thread and panics on failure.
    pub fn open(config: ShmConfig) -> Self {
        let (tx, rx) = mpsc::channel::<InputSample>(config.channel_capacity);
        tokio::task::spawn_blocking(move || run_publisher(rx, config.input_port));
        Self { tx }
    }

    /// Sends an [`InputSample`] into the shared-memory pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`ShmError::ChannelClosed`] if the background thread has
    /// exited.
    pub async fn write(&self, sample: InputSample) -> Result<(), ShmError> {
        self.tx
            .send(sample)
            .await
            .map_err(|_| ShmError::ChannelClosed)
    }

    /// Convenience method: validates and writes raw RGBA bytes as a frame.
    ///
    /// Mirrors `ShmManager.write_raw(data, width, height, frame_id)` on the
    /// Python side. Validates frame dimensions and buffer length before
    /// sending.
    ///
    /// # Parameters
    ///
    /// - `frame_id` — caller-managed frame counter.
    /// - `width` / `height` — frame dimensions in pixels.
    /// - `rgba` — RGBA pixel buffer; length must equal `width * height * 4`.
    ///
    /// # Errors
    ///
    /// - [`ShmError::FrameExceedsMaximum`] — dimensions exceed the maximum.
    /// - [`ShmError::DataLengthMismatch`] — buffer length is wrong.
    /// - [`ShmError::ChannelClosed`] — background thread has exited.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example(writer: &ShmWriter) {
    /// writer.write_raw(0, 1920, 1080, vec![0u8; 1920 * 1080 * 4]).await.unwrap();
    /// # }
    /// ```
    pub async fn write_raw(
        &self,
        frame_id: u64,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> Result<(), ShmError> {
        if width as usize > MAX_WIDTH || height as usize > MAX_HEIGHT {
            return Err(ShmError::FrameExceedsMaximum { width, height });
        }
        let expected = (width * height * 4) as usize;
        if rgba.len() != expected {
            return Err(ShmError::DataLengthMismatch {
                actual: rgba.len(),
                expected,
                width,
                height,
                channels: 4,
            });
        }
        self.write(InputSample {
            frame_id,
            width,
            height,
            rgba,
        })
        .await
    }
}

/// Background publisher loop.
///
/// Receives [`InputSample`]s from the channel and writes each one into shared
/// memory via `loan_uninit()`, avoiding any extra heap allocation. The
/// `timestamp` field is filled automatically with the current Unix timestamp
/// in microseconds.
fn run_publisher(mut rx: mpsc::Receiver<InputSample>, input_port: String) {
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .expect("shm node creation failed");

    let service = node
        .service_builder(&input_port.as_str().try_into().unwrap())
        .publish_subscribe::<InputFrame>()
        .open_or_create()
        .expect("shm input service creation failed");

    let publisher = service.publisher_builder().create().unwrap();
    let rt = tokio::runtime::Handle::current();

    while let Some(sample) = rt.block_on(rx.recv()) {
        // Borrow an uninitialised slot from the shared-memory pool.
        let Ok(mut shm_sample) = publisher.loan_uninit() else {
            continue;
        };

        let expected = (sample.width * sample.height * 4) as usize;
        if sample.rgba.len() < expected {
            continue;
        }

        // Safety: `ptr` points to a valid shared-memory allocation from
        // iceoryx2. Every field is written through `addr_of_mut!`, so no
        // reference to uninitialised memory is ever formed.
        unsafe {
            let ptr = shm_sample.payload_mut().as_mut_ptr();
            addr_of_mut!((*ptr).frame_id).write(sample.frame_id);
            addr_of_mut!((*ptr).timestamp).write(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64,
            );
            addr_of_mut!((*ptr).width).write(sample.width);
            addr_of_mut!((*ptr).height).write(sample.height);
            addr_of_mut!((*ptr).channels).write(4);
            core::ptr::copy_nonoverlapping(
                sample.rgba.as_ptr(),
                addr_of_mut!((*ptr).data) as *mut u8,
                expected,
            );
        }

        // Mark memory as fully initialised and publish the sample.
        let shm_sample = unsafe { shm_sample.assume_init() };
        let _ = shm_sample.send();
    }
}

// ─────────────────────────────────────────────
// ShmReader — read side
// ─────────────────────────────────────────────

/// Async handle for receiving Python-processed frames from iceoryx2 shared
/// memory.
///
/// Mirrors `ShmManager.read()` / `ShmManager.read_one()` on the Python side.
///
/// `ShmReader` is **not** [`Clone`]; it has exclusive ownership of the
/// receiving end. For fan-out scenarios, distribute [`OutputSample`]s at the
/// application layer.
pub struct ShmReader {
    rx: mpsc::Receiver<OutputSample>,
}

impl ShmReader {
    /// Spawns the background subscriber thread and returns a reader handle.
    ///
    /// Returns immediately; iceoryx2 initialisation runs on the background
    /// thread and panics on failure.
    pub fn open(config: ShmConfig) -> Self {
        let (tx, rx) = mpsc::channel::<OutputSample>(config.channel_capacity);
        tokio::task::spawn_blocking(move || {
            run_subscriber(tx, config.output_port, config.poll_interval)
        });
        Self { rx }
    }

    /// Awaits the next processed frame.
    ///
    /// Suspends the current task until a frame is available. Returns `None`
    /// when the background thread has exited and the pipeline is drained.
    ///
    /// Mirrors the `yield` inside `ShmManager.read()` on the Python side.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example(reader: &mut ShmReader) {
    /// while let Some(sample) = reader.read().await {
    ///     println!("frame {}: {}×{}", sample.frame_id, sample.width, sample.height);
    /// }
    /// # }
    /// ```
    pub async fn read(&mut self) -> Option<OutputSample> {
        self.rx.recv().await
    }

    /// Non-blocking frame read; returns immediately.
    ///
    /// Mirrors `ShmManager.read_one()` on the Python side.
    ///
    /// - `Some(sample)` — a frame was available and has been consumed.
    /// - `None` — no frame is ready; try again later.
    ///
    /// Suitable for game loops or render loops where `.await` is undesirable.
    pub fn read_one(&mut self) -> Option<OutputSample> {
        self.rx.try_recv().ok()
    }
}

/// Background subscriber loop.
///
/// Polls `output_port` at `poll_interval`, converts each received
/// [`OutputFrame`] via [`bgr_frame_to_output_sample`], and pushes the result
/// into the channel. If the channel is full or the receiver has been dropped,
/// the remaining frames in the current batch are silently discarded to keep
/// the polling loop running.
fn run_subscriber(tx: mpsc::Sender<OutputSample>, output_port: String, poll_interval: Duration) {
    let node = NodeBuilder::new()
        .create::<ipc::Service>()
        .expect("shm node creation failed");

    let service = node
        .service_builder(&output_port.as_str().try_into().unwrap())
        .publish_subscribe::<OutputFrame>()
        .open_or_create()
        .expect("shm output service creation failed");

    let subscriber = service.subscriber_builder().create().unwrap();

    loop {
        while let Ok(Some(shm_sample)) = subscriber.receive() {
            let sample = bgr_frame_to_output_sample(&*shm_sample);
            // Drop remaining frames in this batch if the channel is full or
            // the receiver has been closed.
            if tx.try_send(sample).is_err() {
                break;
            }
        }
        std::thread::sleep(poll_interval);
    }
}

/// Converts an [`OutputFrame`] (BGR) into an [`OutputSample`] (RGBA, alpha=255).
///
/// Byte-swaps `BGR[b, g, r]` → `RGBA[r, g, b, 255]`, the inverse of the
/// `cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)` call in `ShmManager._decode` on
/// the Python side.
fn bgr_frame_to_output_sample(frame: &OutputFrame) -> OutputSample {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let pixel_count = w * h;

    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for bgr in frame.data[..pixel_count * 3].chunks_exact(3) {
        rgba.push(bgr[2]); // R ← BGR[2]
        rgba.push(bgr[1]); // G ← BGR[1] (unchanged)
        rgba.push(bgr[0]); // B ← BGR[0]
        rgba.push(255); // A = fully opaque
    }

    let mut sample = OutputSample {
        frame_id: frame.frame_id,
        width: frame.width,
        height: frame.height,
        rgba,
        cached_data_url: None,
    };

    sample.cached_data_url = Some(sample.to_data_url());
    sample
}

// ─────────────────────────────────────────────
// ShmManager — unified manager
// ─────────────────────────────────────────────

/// Unified shared-memory manager that owns both the write and read handles.
///
/// Lifecycle methods ([`open`][ShmManager::open] / [`close`][ShmManager::close]
/// / [`is_open`][ShmManager::is_open]) are intentionally symmetric with the
/// Python-side `ShmManager` to simplify cross-language maintenance.
///
/// # Example
///
/// ```rust,no_run
/// #[tokio::main]
/// async fn main() {
///     let mut shm = ShmManager::new(ShmConfig {
///         input_port: "video/input".into(),
///         output_port: "video/output".into(),
///         ..Default::default()
///     });
///
///     shm.open();
///
///     // Write a frame (mirrors Python ShmManager.write_raw)
///     shm.writer().write_raw(0, 1280, 720, vec![0u8; 1280 * 720 * 4]).await.unwrap();
///
///     // Await the result (mirrors Python: for sample in shm.read())
///     if let Some(sample) = shm.reader().read().await {
///         let _url = sample.to_data_url();
///     }
///
///     shm.close();
/// }
/// ```
pub struct ShmManager {
    config: ShmConfig,
    writer: Option<ShmWriter>,
    reader: Option<ShmReader>,
}

impl ShmManager {
    /// Creates a manager instance without connecting to shared memory.
    ///
    /// Call [`open`][ShmManager::open] before reading or writing. Mirrors the
    /// behaviour of the Python `ShmManager.__init__`.
    pub fn new(config: ShmConfig) -> Self {
        Self {
            config,
            writer: None,
            reader: None,
        }
    }

    /// Spawns the background writer and reader threads, establishing the
    /// shared-memory connection.
    ///
    /// Mirrors `ShmManager.open()` on the Python side. Idempotent: calling
    /// `open()` on an already-open manager is a no-op.
    pub fn open(&mut self) {
        if self.writer.is_none() {
            self.writer = Some(ShmWriter::open(self.config.clone()));
        }
        if self.reader.is_none() {
            self.reader = Some(ShmReader::open(self.config.clone()));
        }
    }

    /// Drops the background threads and releases all resources.
    ///
    /// Mirrors `ShmManager.close()` on the Python side. After this call,
    /// [`is_open`][ShmManager::is_open] returns `false`. Calling
    /// [`open`][ShmManager::open] again will re-establish the connection.
    pub fn close(&mut self) {
        self.writer = None;
        self.reader = None;
    }

    /// Returns `true` when the manager is connected and ready to transfer
    /// frames. Mirrors `ShmManager.is_open` on the Python side.
    pub fn is_open(&self) -> bool {
        self.writer.is_some() && self.reader.is_some()
    }

    /// Returns a reference to the write handle.
    ///
    /// # Panics
    ///
    /// Panics if [`open`][ShmManager::open] has not been called.
    pub fn writer(&self) -> &ShmWriter {
        self.writer
            .as_ref()
            .expect("ShmManager not open; call open() first")
    }

    /// Returns a mutable reference to the read handle.
    ///
    /// # Panics
    ///
    /// Panics if [`open`][ShmManager::open] has not been called.
    pub fn reader(&mut self) -> &mut ShmReader {
        self.reader
            .as_mut()
            .expect("ShmManager not open; call open() first")
    }
}
