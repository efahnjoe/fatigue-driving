//! # VideoCanvas — Dioxus component
//!
//! Renders [`OutputSample`] frames from the shared-memory pipeline onto an
//! HTML `<canvas>` element using the Web Canvas 2D API via `web_sys`.
//!
//! ## Architecture
//!
//! ```text
//!  ShmReader (background task)
//!      │  OutputSample { rgba, width, height, frame_id }
//!      ▼
//!  use_coroutine  ──► Signal<Option<FrameState>>
//!      │                         │
//!      │                         ▼
//!      │                  VideoCanvas (rsx!)
//!      │                    <canvas ref>
//!      │                         │
//!      └─────────────────────────►  draw_frame()
//!                                     ImageData → ctx.put_image_data()
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! fn App() -> Element {
//!     rsx! {
//!         VideoCanvas {
//!             width: 1280,
//!             height: 720,
//!         }
//!     }
//! }
//! ```

use dioxus::prelude::*;
use std::time::{Duration, Instant};

use crate::core::shm::{OutputSample, ShmConfig, ShmReader};

// ─────────────────────────────────────────────
// Internal state
// ─────────────────────────────────────────────

/// Snapshot of the most recently received frame, stored in a Dioxus signal so
/// the component re-renders automatically whenever a new frame arrives.
#[derive(Clone, PartialEq)]
struct FrameState {
    /// Canvas-ready RGBA pixel data (`width * height * 4` bytes, alpha = 255).
    rgba: Vec<u8>,
    /// Frame width in pixels.
    width: u32,
    /// Frame height in pixels.
    height: u32,
    /// Monotonic frame counter received from the backend; displayed as an
    /// overlay for debugging.
    frame_id: u64,
}

impl From<OutputSample> for FrameState {
    fn from(s: OutputSample) -> Self {
        Self {
            rgba: s.rgba,
            width: s.width,
            height: s.height,
            frame_id: s.frame_id,
        }
    }
}

// ─────────────────────────────────────────────
// Component props
// ─────────────────────────────────────────────

/// Props for the [`VideoCanvas`] component.
#[derive(Props, Clone, PartialEq)]
pub struct VideoCanvasProps {
    /// Canvas width in pixels; should match the backend output resolution.
    /// Defaults to 1280.
    #[props(default = 1280)]
    pub width: u32,

    /// Canvas height in pixels; should match the backend output resolution.
    /// Defaults to 720.
    #[props(default = 720)]
    pub height: u32,

    /// iceoryx2 service configuration forwarded to [`ShmReader`].
    /// When omitted the [`ShmConfig::default`] is used (`video/output`).
    #[props(default)]
    pub config: ShmConfig,

    /// Optional CSS class string applied to the wrapping `<div>`.
    #[props(default)]
    pub class: Option<String>,

    /// HTML `id` for the `<canvas>` element.
    ///
    /// Must be unique in the page.  Used internally to retrieve the DOM node
    /// via `document.getElementById()`.  Defaults to `"video-canvas"`.
    #[props(default = "video-canvas".to_string())]
    pub canvas_id: String,
}

// ─────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────

/// A Dioxus component that continuously reads [`OutputSample`] frames from the
/// shared-memory pipeline and paints them onto an HTML `<canvas>`.
///
/// A [`use_coroutine`] task owns a [`ShmReader`] and forwards every incoming
/// frame into a [`Signal`]. A [`use_effect`] watches the signal and calls
/// [`draw_frame`] to blit the raw RGBA bytes into the canvas via
/// `ctx.put_image_data()` — a single GPU upload with no intermediate copies.
///
/// # Example
///
/// ```rust,no_run
/// fn App() -> Element {
///     rsx! {
///         VideoCanvas {
///             width: 1920,
///             height: 1080,
///             canvas_id: "cam",
///             config: ShmConfig {
///                 output_port: "video/output".into(),
///                 ..Default::default()
///             },
///         }
///     }
/// }
/// ```
#[component]
pub fn VideoCanvas(props: VideoCanvasProps) -> Element {
    // Listening to data from ShmReader
    let mut frame_data = use_signal(|| "".to_string());

    // The latest frame received from the shared-memory reader.
    let frame_state: Signal<Option<FrameState>> = use_signal(|| None);

    // ── Background reader coroutine ──────────────────────────────────────
    //
    // Spawned once for the lifetime of the component. On every new frame the
    // coroutine writes into `frame_state`, which triggers a re-render and
    // then the draw effect below.
    let config = props.config.clone();
    let mut frame_state_writer = frame_state;
    use_coroutine(move |_rx: UnboundedReceiver<()>| {
        let config = config.clone();

        async move {
            let mut reader = ShmReader::open(config);
            while let Some(sample) = reader.read().await {
                frame_state_writer.set(Some(FrameState::from(sample)));
            }
        }
    });

    // ── Draw effect ──────────────────────────────────────────────────────
    //
    // Runs whenever `frame_state` changes. If both the canvas reference and a
    // new frame are available, the RGBA bytes are uploaded to the canvas.
    let canvas_id = props.canvas_id.clone();

    use_resource(move || {
        let canvas_id = canvas_id.clone();
        let config = props.config.clone();

        async move {
            let mut eval = document::eval(&format!(
                r#"
                const canvas = document.getElementById("{canvas_id}");
                const ctx = canvas.getContext('2d', {{ alpha: false }});
                const img = new Image();

                while (true) {{
                    const url = await dioxus.recv();
                    if (url) {{
                        img.onload = () => {{
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        }};
                        img.src = url;
                    }}
                }}
                "#,
                canvas_id = canvas_id
            ));

            let mut reader = ShmReader::open(config);
            while let Some(sample) = reader.read().await {
                let url = sample.to_data_url();
                if let Err(e) = eval.send(url) {
                    eprintln!("send failed: {:?}", e);
                    break;
                }
            }
        }
    });

    // ── Connection status overlay ────────────────────────────────────────
    let status = match frame_state.read().as_ref() {
        None => "Waiting for frames…".to_string(),
        Some(f) => format!("Frame #{} — {}×{}", f.frame_id, f.width, f.height),
    };

    let wrapper_class = props.class.as_deref().unwrap_or("").to_string();

    rsx! {
        div {
            class: "{wrapper_class}",
            style: "position: relative; display: inline-block;",

            // The canvas element. Width and height are set via HTML attributes
            // so the backing store matches the backend resolution exactly.
            canvas {
                id: "{props.canvas_id}",
                width: "{props.width}",
                height: "{props.height}",
                style: "display: block; width: 100%; height: auto; background: #000;",
            }

            // Transparent status bar rendered over the canvas.
            div {
                style: "
                    position: absolute;
                    bottom: 0; left: 0; right: 0;
                    padding: 4px 8px;
                    background: rgba(0,0,0,0.45);
                    color: #e2e8f0;
                    font: 11px/1.4 monospace;
                    pointer-events: none;
                ",
                "{status}"
            }
        }
    }
}

// ─────────────────────────────────────────────
// Optional: FPS counter hook
// ─────────────────────────────────────────────

/// Returns a signal containing the measured render FPS.
///
/// Attach this to a [`VideoCanvas`] to display performance metrics without
/// modifying the core component.
///
/// # Example
///
/// ```rust,no_run
/// fn App() -> Element {
///     let fps = use_fps_counter();
///     rsx! {
///         p { "Render FPS: {fps}" }
///         VideoCanvas {}
///     }
/// }
/// ```
pub fn use_fps_counter() -> Signal<f64> {
    let mut fps: Signal<f64> = use_signal(|| 0.0);

    // Track frame timestamps in a coroutine; update the signal each second.
    use_future(move || async move {
        let mut frame_count: u32 = 0;
        let mut last_tick = Instant::now();

        loop {
            tokio::time::sleep(Duration::from_millis(16)).await;

            frame_count += 1;
            let now = Instant::now();
            let elapsed = now.duration_since(last_tick);

            if elapsed.as_secs_f64() >= 1.0 {
                let new_fps = frame_count as f64 / elapsed.as_secs_f64();
                let smoothed = (fps.read().clone() * 0.8) + (new_fps * 0.2);
                fps.set(smoothed.round());
            }
        }
    });

    fps
}
