# Fatigue Driving

## Project Overview

This project implements a **Fatigue Driving Analysis Application** using a **client-server architecture** for optimal performance and maintainability.

- **Client**: Built with **Rust + Dioxus**. Responsible for the desktop user interface (video feed display, alerts, settings) and communication with the server.
- **Server**: Built with **Python + FastAPI**, leveraging **OpenCV** for video processing and **ROCm PyTorch/ONNX** for high-performance AI inference on AMD GPUs.

The two components communicate via an HTTP API defined by the server.

## Architecture Summary

| Component  | Language/Framework                    | Role                                          | Hardware Dependency     |
| :--------- | :------------------------------------ | :-------------------------------------------- | :---------------------- |
| **Client** | Rust, Dioxus                          | UI, User Interaction, API Consumption         | None (Standard CPU/RAM) |
| **Server** | Python, FastAPI, OpenCV, PyTorch/ONNX | AI Inference, Model Management, API Provision | **AMD GPU with ROCm**   |

## How to Run

### Prerequisites

- **System**: Linux (AMD GPU ROCm support is best on Linux) or Windows WSL2 with ROCm installed.
- **Client**: [Install Rust](https://www.rust-lang.org/tools/install) and the Dioxus CLI: `cargo binstall dioxus-cli --force`
- **Server**: Python 3.12+, with ROCm drivers properly configured for your AMD GPU.

### Steps

1. **Clone the Repository**

```bash
git clone https://github.com/efahnjoe/fatigue-driving.git

cd fatigue-driving
```

2. **Run scripts to download models**

```bash
bun run postinstall
```

2. **Set up the Server (Python)**

```bash
# In a new terminal, navigate back to the project root
cd server

# if not already installed
pip install uv --break-system-packages

# Installs Python dependencies listed in pyproject.toml
uv sync

# Ensure ROCm is working: python -c "import torch; print(torch.cuda.is_available())" # Should return True if ROCm is mapped correctly
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Set up the Client (Rust)**

```bash
# In a new terminal, navigate back to the project root
cd client

dx serve --platform desktop
```

4. Connect: The client application should automatically connect to http://localhost:8000 to send video frames for analysis.

For detailed development guidelines and technical specifics, see [client/AGENTS.md](client/AGENTS.md) and [server/AGENTS.md](server/AGENTS.md).
