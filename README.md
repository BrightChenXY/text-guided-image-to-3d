# text-guided-image-to-3d

[English](README.md) | [简体中文](README_zh.md)

Gradio demo for text-guided image editing with InstructPix2Pix and remote 3D generation powered by a TRELLIS NVIDIA NIM backend. [Quickstart](#quickstart) on **Linux/WSL**.

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
  

# Quickstart<a id="quickstart"></a>
## Inference
### 1. Install Dependencies
Choose one of the following environment setup options.
#### `conda` setup *(Recommended)*

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
conda env create -f environment.yml
conda activate text-guided-image-to-3d
```

If the environment name inside `environment.yml` is changed locally, activate that exact name or rename it in the file before creating the environment.
This Conda environment is configured for GPU-ready PyTorch with CUDA 12.1 via `pytorch=2.5.1`, `torchvision=0.20.1`, and `pytorch-cuda=12.1`.

#### `uv` setup *(Recommended)*

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
The `pyproject.toml` file is configured so `uv` pulls `torch` and `torchvision` from the official PyTorch CUDA 12.1 index.

#### `pip` setup *(Not Recommended)*

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
The `requirements.txt` file pins the CUDA 12.1 wheels for `torch` and `torchvision`, so this route is also suitable for GPU inference and training.

#### Verify GPU availability

After installing the environment, verify that PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

Expected output on a working GPU setup should show:

- a CUDA-enabled PyTorch build such as `2.5.1+cu121`
- `True` for `torch.cuda.is_available()`
- a CUDA runtime version such as `12.1`
- the name of your NVIDIA GPU


### 2. Deploy the TRELLIS Backend
#### Backend prerequisites

Before starting the container, make sure the backend machine has:

- Docker with NVIDIA GPU support enabled
- An NVIDIA GPU available to the container runtime
- An **NGC API key** for pulling the NIM container
- Internet access for the first startup, so the container can download and warm up the model
- Linux or WSL2 as the host environment

NVIDIA’s Visual GenAI NIM docs require an **NGC personal API key** and use it to authenticate against NVIDIA NGC before pulling the container. NVIDIA also notes that Visual GenAI NIMs can run on **WSL**, and that WSL support is currently in **Public Beta**.

#### Step 1: Export your NGC API key
You can get an API key here: https://build.nvidia.com/microsoft/trellis

```bash
export NGC_API_KEY="<PASTE_YOUR_NGC_API_KEY_HERE>"
```
NVIDIA’s NIM docs use $oauthtoken as the username and the NGC API key as the password for container registry login.

#### Step 2: Login to NVIDIA NGC
```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

#### Step 3: Create a local NIM cache directory
```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 "$LOCAL_NIM_CACHE"
```
The cache directory is mounted into the container so model files and warmup artifacts do not need to be re-downloaded every time. NVIDIA's deploy page and getting-started guide both mount a local cache into `/opt/nim/.cache/`.

#### Step 4: Start the TRELLIS NIM container
```bash
docker run -it --rm --name=nim-server \
  --runtime=nvidia --gpus='"device=0"' \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_VARIANT=large:text+large:image \
  -p 8000:8000 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache/" \
  nvcr.io/nim/microsoft/trellis:latest
```
This starts the TRELLIS NIM service on port `8000`. On first launch, the container downloads the model, initializes the inference pipeline, and performs a warmup step. The general Visual GenAI NIM getting-started guide notes warmup until the logs show `Pipeline warmup: start/done`.
If you want to control which TRELLIS variant is loaded, change parameter `-e NIM_MODEL_VARIANT=<variant>`:
- `base:text`
- `large:text`
- `large:image`
- `large:text+large:image`

#### Step 5: Check that the service is ready
```bash
curl -X GET http://localhost:8000/v1/health/ready
```
A ready server returns:
```JSON
{"status":"ready"}
```
NVIDIA documents `/v1/health/ready` as the readiness check for the running NIM service.

#### Step 6: Test the TRELLIS NIM API
A simple test request looks like this:
```bash
invoke_url="http://localhost:8000/v1/infer"
output_glb_path="result.glb"

response=$(curl -X POST $invoke_url \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "A simple coffee shop interior",
        "seed": 0
      }')

response_body=$(echo "$response" | awk '/{/,EOF-1')
echo $response_body | jq .artifacts[0].base64 | tr -d '"' | base64 --decode > $output_glb_path
```
NVIDIA’s TRELLIS NIM deploy page shows this exact flow: send a request to http://localhost:8000/v1/infer, then decode artifacts[0].base64 into a .glb file.

#### Step 7: Text/Image-to-3D payloads
NVIDIA’s Visual GenAI performance guide shows TRELLIS payloads with:

- `mode: "text"` plus prompt for text-to-3D
- `mode: "image"` plus a base64 image for image-to-3D
- optional sampling controls such as `ss_sampling_steps` and `slat_sampling_steps`

Example image-to-3D request structure:
```bash
input_image_path="input.jpg"
image_b64=$(base64 -w 0 "$input_image_path")

curl -X POST http://localhost:8000/v1/infer \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
        "mode": "image",
        "image": "data:image/png;base64,'${image_b64}'",
        "seed": 0,
        "ss_sampling_steps": 25,
        "slat_sampling_steps": 25
      }'
```
In this repository, that API call is typically wrapped by `pipelines/trellis_client.py`, so your app code does not need to manually construct curl requests during normal usage. The backend URL should be kept in `config.py`. This is an integration detail of this repo; the request structure itself follows NVIDIA’s published TRELLIS NIM examples.

#### Step 8: Point the frontend to the backend
After the NIM container is running, update the TRELLIS backend URL in **`config.py`** so the local Gradio app sends generation requests to the correct host and port.

Typical local deployment:
```Python
TRELLIS_BASE_URL = "http://localhost:8000/v1/infer" # Change it to your API
```
If the backend is running on another Linux/WSL machine, replace localhost with that machine’s reachable IP or hostname.

### 3. Run the Front-end App

Launch the Gradio app with:

```bash
python app.py
```

After launch, Gradio will print a local URL in the terminal. Open it in your browser to use the demo.


## Training

For full training details, including LoRA fine-tuning, local JSONL datasets, Hugging Face online datasets, subset filtering, streaming training, TRELLIS rerank validation, and checkpoint comparison, see [training/README_training.md](training/README_training.md).

### Hugging Face Online Training

The training script supports downloading and training directly from a Hugging Face dataset such as `timbrooks/instructpix2pix-clip-filtered`.

Basic Hugging Face dataset mode:

```bash
accelerate launch training/train_lora_pix2pix.py \
  --dataset-name timbrooks/instructpix2pix-clip-filtered \
  --train-split train \
  --original-image-column original_image \
  --edited-image-column edited_image \
  --edit-prompt-column edit_prompt \
  --output-dir training/outputs/checkpoints/hf_baseline
```

If the dataset does not ship with a validation split, carve one out automatically from the training split:

```bash
accelerate launch training/train_lora_pix2pix.py \
  --dataset-name timbrooks/instructpix2pix-clip-filtered \
  --train-split train \
  --validation-from-train-ratio 0.05 \
  --original-image-column original_image \
  --edited-image-column edited_image \
  --edit-prompt-column edit_prompt \
  --output-dir training/outputs/checkpoints/hf_baseline
```

If you want to train while data is still arriving locally, the project also supports local streaming subset training from `metadata.jsonl` plus filtered indices such as `training/data/final_indices.json`. See [training/README_training.md](training/README_training.md) for the full workflow.

### View Training Curves with TensorBoard

By default, training logs are written to:

```text
<output_dir>/tensorboard
```

For example, if you train with:

```text
--output-dir training/outputs/checkpoints/filtered_stream_lora
```

then the TensorBoard log directory is:

```text
training/outputs/checkpoints/filtered_stream_lora/tensorboard
```

Launch TensorBoard with:

```bash
tensorboard --logdir training/outputs/checkpoints/filtered_stream_lora/tensorboard
```

Then open the local URL shown in the terminal, usually `http://localhost:6006`.

Recommended tags to monitor:

- `train/loss`: raw diffusion training loss.
- `train/learning_rate`: the current optimiser learning rate.
- `val/loss`: validation loss, when validation is enabled.
- `val/previews/*`: saved validation preview strips.
- `trellis/*`: downstream 3D proxy metrics when TRELLIS rerank validation is enabled.

If you reuse the same output directory across multiple runs, TensorBoard will read all event files in that directory. For clean comparisons, use a fresh `--output-dir` or set a dedicated `--tensorboard-log-dir` for each experiment.

# Dependencies / Tech Stack

- Gradio for the web UI.
- PyTorch and torchvision for model execution.
- Diffusers and Transformers for the image editing pipeline.
- InstructPix2Pix for text-guided image editing.
- TRELLIS for backend 3D generation.
- NumPy, Pillow, Requests, and related utility libraries for image handling and API communication.



# Notes

- The local app handles image preprocessing and InstructPix2Pix editing, while 3D generation is sent to the configured TRELLIS backend service.
- The TRELLIS endpoint is defined in `config.py`; update it if your backend address or port changes.
- Generated files are written under `outputs/`, including edited images and exported `.glb` assets.
- Dependency versions are pinned through `requirements.txt`, `environment.yml`, and `pyproject.toml` to keep the setup aligned with Python 3.10.


# License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.
