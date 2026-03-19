mod components;
mod core;

use components::source_selector::SourceSelector;
use components::video_canvas::{VideoCanvas,use_fps_counter};
use dioxus::prelude::*;
use dxc::prelude::*;

const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");
const TAILWIND_CSS: Asset = asset!("/assets/tailwind.css");

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let fps = use_fps_counter();

    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        document::Link { rel: "stylesheet", href: DXC_THEMES }

        DxcContainer {
            DxcMain {
                div {
                    class: "flex flex-col gap-4",
                    h1 { "疲劳驾驶分析" }

                    div {
                        p { "Render FPS: {fps}" }
                        
                        SourceSelector{}
                        VideoCanvas {
                            canvas_id: "videoCanvas",
                        }
                    }
                }
            }
        }
    }
}
