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
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        document::Link { rel: "stylesheet", href: TAILWIND_CSS }
        document::Link { rel: "stylesheet", href: DXC_THEMES }

        DxcContainer {
            h1 { "疲劳驾驶分析" }

            DxcMain {
                div {
                    DxcButton{"启动摄像头"}
                    DxcButton{"停止摄像头"}
                    DxcButton{"开启疲劳检测"}
                    DxcButton{"上传文件"}
                    DxcInput{
                        type_: "file",
                    }
                }

                div {
                    canvas {
                        id: "mainCanvas",
                        width: "640",
                        height: "480",
                    }
                    div {
                        id:"placeholderText",
                        "摄像头或文件显示区域"
                    }
                }


                div {"系统就绪"}
            }
        }
    }
}
