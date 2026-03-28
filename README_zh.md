# text-guided-image-to-3d

[English](README.md) | [简体中文]

一个基于 Gradio 的演示项目，用于通过 InstructPix2Pix 进行文本引导图像编辑，并调用远程 TRELLIS 后端生成 3D 模型。

## Features

- 上传参考图片，并结合文本提示进行编辑。
- 在生成 3D 之前先预览编辑后的图片效果。
- 通过远程 TRELLIS 后端生成 `.glb` 资产。
- 可在应用内直接下载生成后的模型文件。
- 中间结果会保存在本地 `outputs/` 目录，便于查看和调试。

## Project Structure

- `app.py`: Gradio 界面与整体流程编排。
- `config.py`: 运行配置、输出目录、模型 ID 与 TRELLIS 接口地址。
- `pipelines/`: 图像预处理、InstructPix2Pix 编辑与 TRELLIS 客户端逻辑。
- `assets/`: 静态资源目录。
- `outputs/`: 保存编辑后的图片、预览文件和生成的网格模型。
- `requirements.txt`、`environment.yml`、`pyproject.toml`: 依赖与环境定义文件。

## Requirements

- 仅支持 Python `>=3.10,<3.11`，也就是必须使用 Python 3.10。
- 需要安装 Git 用于克隆仓库。
- 需要能够联网下载模型依赖，并访问已配置的 TRELLIS 后端服务。
- 建议使用支持 CUDA 的 GPU 以获得更快的本地 InstructPix2Pix 推理速度，CPU 也可运行但会更慢。

## Quickstart

### conda setup

```bash
git clone https://github.com/BrightChenXY/text-guided-image-to-3d.git
cd text-guided-image-to-3d
conda env create -f environment.yml
conda activate text-guided-image-to-3d
```

如果你本地修改了 `environment.yml` 里的环境名称，请激活修改后的实际名称，或者先把文件中的名称改回再创建环境。

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

这个仓库按非打包应用方式配置了 `uv`（`tool.uv.package = false`），因此优先推荐使用 `uv sync`。如果你更习惯基于 `requirements.txt` 的流程，或者本地 `uv` 版本在当前结构下不适合使用 `sync`，也可以改用 `uv pip install -r requirements.txt`。

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

请确认当前环境中的 `python` 指向 Python 3.10。


## Running the App

使用以下命令启动 Gradio 应用：

```bash
python app.py
```

默认情况下，应用会运行在 `0.0.0.0:7860`。

## Dependencies / Tech Stack

- Gradio：Web 界面。
- PyTorch 与 torchvision：模型推理执行。
- Diffusers 与 Transformers：图像编辑流程的核心依赖。
- InstructPix2Pix：文本引导图像编辑。
- TRELLIS：后端 3D 生成服务。
- NumPy、Pillow、Requests 等工具库：用于图像处理和接口通信。

## Notes

- 本地应用负责图像预处理和 InstructPix2Pix 编辑，3D 生成功能会请求已配置的远程 TRELLIS 服务。
- TRELLIS 接口地址定义在 `config.py` 中；如果后端地址或端口变化，请同步更新配置。
- 生成结果会写入 `outputs/` 目录，包括编辑后的图片和导出的 `.glb` 文件。
- `requirements.txt`、`environment.yml` 和 `pyproject.toml` 已对依赖范围做了约束，以保持与 Python 3.10 环境一致。

## License

暂未添加许可证信息。
