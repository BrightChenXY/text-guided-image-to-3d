# text-guided-image-to-3d

[English](README.md) | [简体中文](README_zh.md)

一个基于 Gradio 的演示项目，支持通过 InstructPix2Pix 进行文本引导图像编辑，并通过 TRELLIS NVIDIA NIM 远程后端生成 3D 结果。在 Linux/WSL 上部署，可直接查看 [Quickstart](#quickstart)。

## Features

- 上传参考图片，并结合文本提示进行编辑。
- 在生成 3D 之前先预览编辑后的图片效果。
- 通过远程 TRELLIS 后端生成 `.glb` 资产。
- 可在应用内直接下载生成后的模型文件。
- 中间结果会保存在本地 `outputs/` 目录，便于查看和调试。

## Project Structure

```text
text-guided-image-to-3d/
├── app.py
├── config.py
├── pipelines/
│   ├── image_editor.py
│   ├── mock_backend.py
│   ├── preprocess.py
│   ├── text_to_image.py
│   └── trellis_client.py
├── training/
│   ├── data/
│   ├── outputs/
│   ├── dataset.py
│   ├── eval_trellis_compare.py
│   ├── infer_lora_pix2pix.py
│   ├── prepare_metadata.py
│   ├── split_filtered_metadata.py
│   ├── train_lora_pix2pix.py
│   ├── trellis_eval.py
│   ├── README_training.md
│   └── README_training_zh.md
├── assets/
│   ├── demo_templates.json
│   ├── placeholder.glb
│   └── template/
│       ├── edited_imgs/
│       ├── input_imgs/
│       └── output_glbs/
├── outputs/
│   ├── edited/
│   ├── meshes/
│   └── previews/
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── README.md
└── README_zh.md
```

关键文件说明：

- `app.py`：主 Gradio 入口，负责串联图像预处理、前端编辑、TRELLIS 请求、演示模板和结果展示。
- `config.py`：集中管理运行配置，包括模型 ID、TRELLIS 接口地址、默认生成参数、输出目录以及可选的 LoRA 设置。
- `pipelines/image_editor.py`：加载 InstructPix2Pix 编辑器并执行文本引导的图像编辑，推理时也支持可选的 LoRA 增强。
- `pipelines/trellis_client.py`：封装远程 TRELLIS API 调用，并把返回的 `.glb` 资产保存到本地输出目录。
- `training/train_lora_pix2pix.py`：LoRA 训练主脚本，支持本地 JSONL 数据集、Hugging Face 在线数据集、TensorBoard 日志、checkpoint 保存和 TRELLIS rerank 验证。
- `training/dataset.py`：训练阶段共用的数据集与预处理工具，覆盖本地 metadata、Hugging Face 数据集、过滤逻辑和流式子集训练。
- `training/trellis_eval.py`：黑盒 TRELLIS proxy scoring 工具，用于通过下游 3D 友好指标评估编辑后的图像。
- `training/eval_trellis_compare.py`：离线对比脚本，用 TRELLIS proxy metrics 比较 baseline 和 LoRA 增强模型，并输出对比图表。
- `assets/demo_templates.json`：预置演示模板清单，记录缓存的输入图、编辑结果预览和可选 GLB 输出。
- `outputs/`：默认本地输出根目录，用于保存编辑图、预览图、生成网格以及中间结果。

## Requirements

- 仅支持 Python `>=3.10,<3.11`，也就是必须使用 Python 3.10。
- 需要安装 Git 用于克隆仓库。
- 需要能够联网下载模型依赖，并访问已配置的 TRELLIS 后端服务。
- 建议使用支持 CUDA 的 GPU 以获得更快的本地 InstructPix2Pix 推理速度，CPU 也可运行但会更慢。

# Quickstart<a id="quickstart"></a>
## ① 安装依赖

### `conda` setup *(推荐)*

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
conda env create -f environment.yml
conda activate text-guided-image-to-3d
```

如果你本地修改了 `environment.yml` 里的环境名称，请激活修改后的实际名称，或者先把文件中的名称改回再创建环境。
这个 Conda 环境已经按 GPU 路线配置好了 `pytorch=2.5.1`、`torchvision=0.20.1` 和 `pytorch-cuda=12.1`。

### `uv` setup *(推荐)*

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
uv venv --python 3.10
source .venv/bin/activate
# Windows alternative
# .venv\Scripts\activate
uv sync
```

这个仓库按非打包应用方式配置了 `uv`（`tool.uv.package = false`），因此优先推荐使用 `uv sync`。如果你更习惯基于 `requirements.txt` 的流程，或者本地 `uv` 版本在当前结构下不适合使用 `sync`，也可以改用 `uv pip install -r requirements.txt`。
`pyproject.toml` 已经配置为让 `uv` 从官方 PyTorch CUDA 12.1 索引拉取 `torch` 和 `torchvision`。

### `pip` setup *(不推荐)*

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
python -m venv .venv
source .venv/bin/activate
# Windows alternative
# .venv\Scripts\activate
pip install -r requirements.txt
```

请确认当前环境中的 `python` 指向 Python 3.10。
`requirements.txt` 现在也固定到了 CUDA 12.1 的 `torch` 和 `torchvision` wheel，因此这条路径同样可以用于 GPU 推理和训练。

### 验证 GPU 是否可用

环境安装完成后，可以用下面的命令确认 PyTorch 是否已经识别到 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

正常情况下，输出里应当包含：

- 形如 `2.5.1+cu121` 的 CUDA 版 PyTorch
- `torch.cuda.is_available()` 为 `True`
- 类似 `12.1` 的 CUDA runtime 版本
- 你的 NVIDIA GPU 名称

## ② 部署 TRELLIS 后端

### 后端前置条件

在启动容器之前，请确认后端机器具备以下条件：

- 已安装 Docker，并启用了 NVIDIA GPU 支持。
- 容器运行时可以访问 NVIDIA GPU。
- 拥有用于拉取 NIM 容器的 **NGC API key**。
- 首次启动时可以联网，以便容器下载模型并完成预热。
- 宿主环境为 Linux 或 WSL2。

NVIDIA 的 Visual GenAI NIM 文档要求使用 **NGC personal API key**，并在拉取容器前通过 NVIDIA NGC 完成认证。文档同时说明 Visual GenAI NIM 可以运行在 **WSL** 上，而 WSL 支持目前处于 **Public Beta** 阶段。

### Step 1: 导出 NGC API key

你可以在这里获取 API key：https://build.nvidia.com/microsoft/trellis

```bash
export NGC_API_KEY="<PASTE_YOUR_NGC_API_KEY_HERE>"
```

NVIDIA 的 NIM 文档使用 `$oauthtoken` 作为用户名，并将 NGC API key 作为容器镜像仓库登录密码。

### Step 2: 登录 NVIDIA NGC

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Step 3: 创建本地 NIM 缓存目录

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 "$LOCAL_NIM_CACHE"
```

该缓存目录会挂载到容器内，这样模型文件和预热产物就不需要每次重新下载。NVIDIA 的部署页面和快速开始文档都建议将本地缓存挂载到 `/opt/nim/.cache/`。

### Step 4: 启动 TRELLIS NIM 容器

```bash
docker run -it --rm --name=nim-server \
  --runtime=nvidia --gpus='"device=0"' \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_VARIANT=large:text+large:image \
  -p 8000:8000 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache/" \
  nvcr.io/nim/microsoft/trellis:latest
```

这会在 `8000` 端口启动 TRELLIS NIM 服务。首次启动时，容器会下载模型、初始化推理流程并执行预热。Visual GenAI NIM 的通用快速开始文档提到，只有在日志出现 `Pipeline warmup: start/done` 后，服务才算真正就绪。

如果你想控制加载的 TRELLIS 变体，可以修改参数 `-e NIM_MODEL_VARIANT=<variant>`：

- `base:text`
- `large:text`
- `large:image`
- `large:text+large:image`

### Step 5: 检查服务是否就绪

```bash
curl -X GET http://localhost:8000/v1/health/ready
```

服务就绪后通常会返回：

```json
{"status":"ready"}
```

NVIDIA 文档将 `/v1/health/ready` 作为正在运行的 NIM 服务的就绪检查接口。

### Step 5: 测试 TRELLIS NIM API

一个简单的测试请求如下：

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

NVIDIA 的 TRELLIS NIM 部署页面展示了这一流程：向 `http://localhost:8000/v1/infer` 发送请求，再将 `artifacts[0].base64` 解码为 `.glb` 文件。

### Step 6: Text/Image-to-3D 请求体

NVIDIA 的 Visual GenAI 性能指南展示了 TRELLIS 请求体中常见的字段：

- `mode: "text"`，配合 `prompt` 用于 text-to-3D。
- `mode: "image"`，配合 base64 图片用于 image-to-3D。
- 可选采样控制参数，例如 `ss_sampling_steps` 和 `slat_sampling_steps`。

示例 image-to-3D 请求结构如下：

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

在这个仓库里，这类 API 调用通常封装在 `pipelines/trellis_client.py` 中，因此正常使用时不需要手动拼接 `curl` 请求。后端地址应保持在 `config.py` 中统一管理。这里主要是说明本仓库的集成方式，请求结构本身遵循 NVIDIA 官方公开的 TRELLIS NIM 示例。

### Step 7: 让前端连接后端

在 NIM 容器启动后，请更新 **`config.py`** 中的 TRELLIS 后端地址，确保本地 Gradio 应用把生成请求发送到正确的主机和端口。

典型的本地部署配置如下：

```python
TRELLIS_BASE_URL = "http://localhost:8000/v1/infer" # Change it to your API
```

如果后端运行在另一台 Linux 或 WSL 机器上，请将 `localhost` 替换为该机器可访问的 IP 或主机名。

## ③ 运行前端应用

使用以下命令启动 Gradio 应用：

```bash
python app.py
```

启动后，Gradio 会在终端输出本地访问地址。用浏览器打开即可体验演示。

# 训练

更完整的训练说明，包括 LoRA 微调、本地 JSONL 数据集、Hugging Face 在线数据集、子集过滤、流式训练、TRELLIS rerank 验证和 checkpoint 对比，请查看 [training/README_training_zh.md](training/README_training_zh.md)。

## Hugging Face 在线训练

训练脚本支持直接从 Hugging Face 数据集下载并训练，例如 `timbrooks/instructpix2pix-clip-filtered`。

基础 Hugging Face 数据集模式：

```bash
accelerate launch training/train_lora_pix2pix.py \
  --dataset-name timbrooks/instructpix2pix-clip-filtered \
  --train-split train \
  --original-image-column original_image \
  --edited-image-column edited_image \
  --edit-prompt-column edit_prompt \
  --output-dir training/outputs/checkpoints/hf_baseline
```

如果数据集本身没有单独的验证集 split，可以从训练集里自动切出一部分作为验证集：

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

如果你想在本地图片还在持续补充时边准备数据边训练，项目也支持基于 `metadata.jsonl` 和 `training/data/final_indices.json` 的本地流式子集训练。完整流程请见 [training/README_training_zh.md](training/README_training_zh.md)。

## 使用 TensorBoard 查看训练过程

默认情况下，训练日志会写到：

```text
<output_dir>/tensorboard
```

例如如果你的训练命令里使用了：

```text
--output-dir training/outputs/checkpoints/filtered_stream_lora
```

那么 TensorBoard 日志目录就是：

```text
training/outputs/checkpoints/filtered_stream_lora/tensorboard
```

可以用下面的命令启动 TensorBoard：

```bash
tensorboard --logdir training/outputs/checkpoints/filtered_stream_lora/tensorboard
```

然后在浏览器中打开终端里显示的本地地址，通常是 `http://localhost:6006`。

建议重点观察这些标签：

- `train/loss`：原始 diffusion 训练损失。
- `train/learning_rate`：当前优化器学习率。
- `val/loss`：启用验证时的验证集损失。
- `val/previews/*`：保存下来的验证预览图。
- `trellis/*`：启用 TRELLIS rerank 验证后写入的下游 3D proxy 指标。

如果你多次复用同一个输出目录，TensorBoard 会把该目录下的所有事件文件一起读取。为了更干净地比较实验，建议为每次实验使用新的 `--output-dir`，或者显式指定单独的 `--tensorboard-log-dir`。

# Dependencies / Tech Stack

- Gradio：Web 界面。
- PyTorch 与 torchvision：模型推理执行。
- Diffusers 与 Transformers：图像编辑流程的核心依赖。
- InstructPix2Pix：文本引导图像编辑。
- TRELLIS：后端 3D 生成。
- NumPy、Pillow、Requests 等工具库：用于图像处理和接口通信。

# Notes

- 本地应用负责图像预处理和 InstructPix2Pix 编辑，3D 生成功能会请求已配置的远程 TRELLIS 服务。
- TRELLIS 接口地址定义在 `config.py` 中；如果后端地址或端口变化，请同步更新配置。
- 生成结果会写入 `outputs/` 目录，包括编辑后的图片和导出的 `.glb` 文件。
- `requirements.txt`、`environment.yml` 和 `pyproject.toml` 已对依赖范围做了约束，以保持与 Python 3.10 环境一致。

# 许可证

本项目基于 MIT License 发布。完整许可证内容请查看 [LICENSE](LICENSE)。
