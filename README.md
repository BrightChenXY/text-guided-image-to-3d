# text-guided-image-to-3d

[English] | [简体中文](README_zh.md)

Gradio demo for text-guided image editing with InstructPix2Pix and remote TRELLIS-based 3D generation.

## Features

- Upload a reference image and refine it with a text prompt.
- Preview the edited image before starting 3D generation.
- Generate a `.glb` asset through a remote TRELLIS backend.
- Download the generated model directly from the app.
- Keep intermediate outputs in local `outputs/` directories for inspection.

## Project Structure

- `app.py`: Gradio interface and end-to-end workflow orchestration.
- `config.py`: Runtime settings, output directories, model IDs, and TRELLIS endpoint configuration.
- `pipelines/`: Image preprocessing, InstructPix2Pix editing, and TRELLIS client logic.
- `assets/`: Static project assets.
- `outputs/`: Saved edited images, previews, and generated meshes.
- `requirements.txt`, `environment.yml`, `pyproject.toml`: Dependency and environment definitions.

## Requirements

- Python `>=3.10,<3.11` only. Python 3.10 is required for this project.
- Git for cloning the repository.
- Network access to download model dependencies and reach the configured TRELLIS backend.
- A CUDA-capable GPU is recommended for faster local InstructPix2Pix inference, though CPU execution is possible.

## Quickstart

### conda setup

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
conda env create -f environment.yml
conda activate text-guided-image-to-3d
```

If the environment name inside `environment.yml` is changed locally, activate that exact name or rename it in the file before creating the environment.

### uv setup

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
uv venv --python 3.10
source .venv/bin/activate
# Windows alternative
# .venv\Scripts\activate
uv sync
```

This repository is configured as a non-packaged app (`tool.uv.package = false`), so `uv sync` is the preferred workflow. If you prefer a requirements-based flow or your local `uv` setup does not use `sync` here, you can fall back to `uv pip install -r requirements.txt`.

### pip setup

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
python -m venv .venv
source .venv/bin/activate
# Windows alternative
# .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure the `python` executable in the environment is Python 3.10.

## Running the App

Launch the Gradio app with:

```bash
python app.py
```

By default, the app starts on `0.0.0.0:7860`.

## Dependencies / Tech Stack

- Gradio for the web UI.
- PyTorch and torchvision for model execution.
- Diffusers and Transformers for the image editing pipeline.
- InstructPix2Pix for text-guided image editing.
- TRELLIS for backend 3D generation.
- NumPy, Pillow, Requests, and related utility libraries for image handling and API communication.

## Notes

- The local app handles image preprocessing and InstructPix2Pix editing, while 3D generation is sent to the configured TRELLIS backend service.
- The TRELLIS endpoint is defined in `config.py`; update it if your backend address or port changes.
- Generated files are written under `outputs/`, including edited images and exported `.glb` assets.
- Dependency versions are pinned through `requirements.txt`, `environment.yml`, and `pyproject.toml` to keep the setup aligned with Python 3.10.

## License

License information has not been added yet.
