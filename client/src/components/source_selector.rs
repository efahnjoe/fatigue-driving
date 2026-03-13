//! # SourceSelector — Dioxus component
//!
//! Toolbar for switching between three input sources (camera, image file, video
//! file) and streaming the resulting RGBA frames into the shared-memory
//! pipeline via [`ShmWriter`].
//!
//! ## Data flow
//!
//! ```text
//! ┌──────────────┐  RGBA frames   ┌───────────────┐  InputFrame (SHM)
//! │ SourceSelector│ ─────────────► │   ShmWriter   │ ─────────────────► Python
//! │  (UI thread) │                │ (background)  │
//! └──────────────┘                └───────────────┘
//!        │
//!  ┌─────┼──────────┐
//!  ▼     ▼          ▼
//! Camera  Image     Video
//! nokhwa  image     opencv / ffmpeg
//! ```

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use dioxus::prelude::*;
use dxc::prelude::*;
use dxc_icons::{Picture, VideoCamera, VideoPlay};
// use image::GenericImageView;

use crate::core::shm::{ShmConfig, ShmWriter};

// ─────────────────────────────────────────────
// Input source enum
// ─────────────────────────────────────────────

/// Which media source is currently active.
#[derive(Debug, Clone, PartialEq)]
pub enum InputSource {
    /// Live camera feed streamed frame-by-frame.
    Camera,
    /// A single static image sent as one frame.
    Image,
    /// A video file decoded and streamed frame-by-frame.
    Video,
}

// ─────────────────────────────────────────────
// Props
// ─────────────────────────────────────────────

/// Props for the [`SourceSelector`] component.
#[derive(Props, Clone)]
pub struct SourceSelectorProps {
    /// iceoryx2 service configuration used to open the [`ShmWriter`].
    /// Falls back to [`ShmConfig::default`] (`video/input`, 1 ms poll).
    #[props(default)]
    pub config: ShmConfig,

    /// Optional callback fired whenever the active [`InputSource`] changes.
    /// Receives the new source value so parent components can react.
    #[props(default)]
    pub on_source_change: Option<EventHandler<InputSource>>,
}

impl PartialEq for SourceSelectorProps {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config // skip Signal / Sender fields
    }
}

// ─────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────

/// Toolbar with three source buttons (camera / image / video).
///
/// Each button updates the `current` signal and immediately begins pushing
/// RGBA frames to the SHM pipeline:
///
/// - **Camera** — opens device 0 via `nokhwa`, captures frames in a background
///   task, and stops when the source changes.
/// - **Image** — opens a native file-picker (`rfd`), loads the selected file
///   with the `image` crate, and sends a single frame.
/// - **Video** — opens a native file-picker (`rfd`), decodes the video with
///   `opencv`, and streams frames until the file ends or the source changes.
#[component]
pub fn SourceSelector(props: SourceSelectorProps) -> Element {
    let mut current: Signal<InputSource> = use_signal(|| InputSource::Camera);

    // Construct the ShmWriter exactly once for this component instance.
    // `use_hook` runs only on the first render, so the background publisher
    // thread is started once and reused across re-renders.
    let writer = use_hook(|| ShmWriter::open(props.config.clone()));

    // A shared flag that cancels any running camera / video loop when the
    // user switches to a different source.
    let running: Signal<Arc<AtomicBool>> = use_signal(|| Arc::new(AtomicBool::new(false)));

    // ── Helper: stop any active streaming loop ───────────────────────────
    let stop_current = move || {
        running.read().store(false, Ordering::Relaxed);
    };

    rsx! {
        div { class: "flex gap-3 p-4 bg-gray-900 border-b border-gray-700",

            // ── Camera button ─────────────────────────────────────────────
            {
                let writer = writer.clone();
                let mut running = running;
                rsx! {
                    SourceButton {
                        active: *current.read() == InputSource::Camera,
                        label: "Camera",
                        icon: rsx! { DxcIcon { VideoCamera {} } },
                        onclick: move |_| {
                            stop_current();
                            current.set(InputSource::Camera);
                            if let Some(ref cb) = props.on_source_change { cb.call(InputSource::Camera); }

                            // Start a new flag for this session
                            let flag = Arc::new(AtomicBool::new(true));
                            running.set(flag.clone());

                            let writer = writer.clone();
                            spawn(async move {
                                if let Err(e) = stream_camera(writer, flag).await {
                                    tracing::error!("camera error: {}", e);
                                }
                            });
                        },
                    }
                }
            }

            // ── Image button ──────────────────────────────────────────────
            {
                let writer = writer.clone();
                rsx! {
                    SourceButton {
                        active: *current.read() == InputSource::Image,
                        label: "Image",
                        icon: rsx! { DxcIcon { Picture {} } },
                        onclick: move |_| {
                            stop_current();
                            current.set(InputSource::Image);
                            if let Some(ref cb) = props.on_source_change { cb.call(InputSource::Image); }

                            let writer = writer.clone();
                            spawn(async move {
                                let Some(handle) = rfd::AsyncFileDialog::new()
                                    .add_filter("Image", &["png", "jpg", "jpeg", "bmp", "webp"])
                                    .pick_file()
                                    .await
                                else {
                                    return;
                                };

                                let path = handle.path().to_owned();
                                if let Err(e) = send_image(writer, path).await {
                                    tracing::error!("image error: {}", e);
                                }
                            });
                        },
                    }
                }
            }

            // ── Video button ──────────────────────────────────────────────
            {
                let writer = writer.clone();
                let mut running = running;
                rsx! {
                    SourceButton {
                        active: *current.read() == InputSource::Video,
                        label: "Video",
                        icon: rsx! { DxcIcon { VideoPlay {} } },
                        onclick: move |_| {
                            stop_current();
                            current.set(InputSource::Video);
                            if let Some(ref cb) = props.on_source_change { cb.call(InputSource::Video); }

                            let flag = Arc::new(AtomicBool::new(true));
                            running.set(flag.clone());

                            let writer = writer.clone();
                            spawn(async move {
                                let Some(handle) = rfd::AsyncFileDialog::new()
                                    .add_filter("Video", &["mp4", "avi", "mov", "mkv", "webm"])
                                    .pick_file()
                                    .await
                                else {
                                    return;
                                };

                                let path = handle.path().to_owned();
                                if let Err(e) = stream_video(writer, path, flag).await {
                                    tracing::error!("video error: {}", e);
                                }
                            });
                        },
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// Internal: shared button sub-component
// ─────────────────────────────────────────────

/// Re-usable styled button used by each source option.
///
/// Applies `bg-blue-600` when `active`, otherwise a muted gray style.
#[derive(Props, Clone, PartialEq)]
struct SourceButtonProps {
    active: bool,
    label: &'static str,
    icon: Element,
    onclick: EventHandler<MouseEvent>,
}

#[component]
fn SourceButton(props: SourceButtonProps) -> Element {
    let class = if props.active {
        "px-4 py-2 rounded-lg text-sm font-medium transition-colors \
         bg-blue-600 text-white flex items-center gap-2"
    } else {
        "px-4 py-2 rounded-lg text-sm font-medium transition-colors \
         bg-gray-700 text-gray-300 hover:bg-gray-600 flex items-center gap-2"
    };

    rsx! {
        DxcButton {
            class: "{class}",
            onclick: move |e| props.onclick.call(e),
            {props.icon}
            span { "{props.label}" }
        }
    }
}

// ─────────────────────────────────────────────
// Source drivers
// ─────────────────────────────────────────────

/// Captures frames from the default camera (device index 0) and sends them
/// to the SHM pipeline until `running` is set to `false`.
///
/// Uses `nokhwa` for cross-platform camera access.  Each captured frame is
/// decoded to RGBA and forwarded via [`ShmWriter::write_raw`].
async fn stream_camera(writer: ShmWriter, running: Arc<AtomicBool>) -> anyhow::Result<()> {
    use nokhwa::{
        Camera,
        pixel_format::RgbAFormat,
        utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    };

    // Camera initialisation and capture run on a blocking thread so the
    // Tokio runtime is not stalled by synchronous I/O.
    tokio::task::spawn_blocking(move || {
        let index = CameraIndex::Index(0);
        let format =
            RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(index, format)?;
        camera.open_stream()?;

        let rt = tokio::runtime::Handle::current();
        let mut frame_id: u64 = 0;

        while running.load(Ordering::Relaxed) {
            let frame = camera.frame()?;
            let decoded = frame.decode_image::<RgbAFormat>()?;
            let (w, h) = (decoded.width(), decoded.height());
            let rgba = decoded.into_raw();

            rt.block_on(writer.write_raw(frame_id, w, h, rgba))?;
            frame_id += 1;
        }

        camera.stop_stream()?;
        Ok::<_, anyhow::Error>(())
    })
    .await??;

    Ok(())
}

/// Loads an image file, converts it to RGBA, and sends a single frame.
///
/// The `image` crate handles format detection automatically (PNG, JPEG, BMP,
/// WebP, …).
async fn send_image(writer: ShmWriter, path: std::path::PathBuf) -> anyhow::Result<()> {
    // Decoding is CPU-bound; keep it off the async executor.
    let (w, h, rgba) = tokio::task::spawn_blocking(move || {
        let img = image::open(&path)?.to_rgba8();
        let (w, h) = img.dimensions();
        Ok::<_, anyhow::Error>((w, h, img.into_raw()))
    })
    .await??;

    writer.write_raw(0, w, h, rgba).await?;
    Ok(())
}

/// Decodes a video file frame-by-frame using `ffmpeg-next` and streams each
/// frame to the SHM pipeline until the file ends or `running` is set to
/// `false`.
///
/// `ffmpeg-next` wraps `libavcodec` / `libavformat` which are available on
/// every major Linux distro:
///
/// ```bash
/// # Ubuntu / Debian
/// sudo apt install pkg-config libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev
/// sudo apt install \
/// ```
///
/// Frames are decoded to `AV_PIX_FMT_RGBA` via `swscale` before being sent
/// through the SHM pipeline.  The loop is throttled to the stream's native
/// frame rate to avoid overloading the pipeline.
async fn stream_video(
    writer: ShmWriter,
    path: std::path::PathBuf,
    running: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    // ffmpeg is synchronous — keep it off the async executor.
    tokio::task::spawn_blocking(move || {
        use ffmpeg::format::Pixel;
        use ffmpeg::software::scaling::{context::Context as SwsContext, flag::Flags};
        use ffmpeg::util::frame::video::Video as AvFrame;
        use ffmpeg_next as ffmpeg;

        ffmpeg::init()?;

        // Open the input file and locate the best video stream.
        let mut ictx = ffmpeg::format::input(&path)?;
        let video_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| anyhow::anyhow!("no video stream found in {:?}", path))?;

        let stream_index = video_stream.index();

        // Read the native frame rate for throttling (falls back to 30 fps).
        let time_base = video_stream.time_base();
        let avg_frame_rate = video_stream.avg_frame_rate();
        let fps = avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator().max(1) as f64;
        let frame_delay = std::time::Duration::from_secs_f64(1.0 / fps.max(1.0));

        // Build the decoder for the video stream.
        let context_decoder =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let mut decoder = context_decoder.decoder().video()?;

        // Build a swscale converter: any pixel format → RGBA (packed).
        let mut scaler = SwsContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGBA,
            decoder.width(),
            decoder.height(),
            Flags::BILINEAR,
        )?;

        let rt = tokio::runtime::Handle::current();
        let mut frame_id: u64 = 0;
        let w = decoder.width();
        let h = decoder.height();

        'packet_loop: for (stream, packet) in ictx.packets() {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            if stream.index() != stream_index {
                continue;
            }

            decoder.send_packet(&packet)?;

            let mut decoded = AvFrame::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if !running.load(Ordering::Relaxed) {
                    break 'packet_loop;
                }

                // Convert the decoded frame to RGBA.
                let mut rgba_frame = AvFrame::empty();
                scaler.run(&decoded, &mut rgba_frame)?;

                // `data(0)` is the first (and only) plane of a packed RGBA frame.
                let rgba: Vec<u8> = rgba_frame.data(0).to_vec();
                rt.block_on(writer.write_raw(frame_id, w, h, rgba))?;
                frame_id += 1;

                std::thread::sleep(frame_delay);
            }
        }

        // Flush any frames buffered inside the decoder.
        decoder.send_eof()?;
        let mut decoded = AvFrame::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            let mut rgba_frame = AvFrame::empty();
            scaler.run(&decoded, &mut rgba_frame)?;
            let rgba: Vec<u8> = rgba_frame.data(0).to_vec();
            rt.block_on(writer.write_raw(frame_id, w, h, rgba))?;
            frame_id += 1;
        }

        Ok::<_, anyhow::Error>(())
    })
    .await??;

    Ok(())
}
