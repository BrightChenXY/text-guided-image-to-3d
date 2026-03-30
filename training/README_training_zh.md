# InstructPix2Pix LoRA 训练说明

[English](README_training.md) | [简体中文](README_training_zh.md)

本目录提供一套最小可执行的 `timbrooks/instruct-pix2pix` LoRA 微调与推理管线，用来提升你当前前端图像编辑阶段对 TRELLIS `large:image` 的适配性。

## 目录建议

```text
training/
  data/
    custom_train/
      images/
      metadata.jsonl
    custom_val/
      images/
      metadata.jsonl
  outputs/
    checkpoints/
    samples/
  dataset.py
  prepare_metadata.py
  train_lora_pix2pix.py
  infer_lora_pix2pix.py
  README_training.md
  README_training_zh.md
```

## 1. 数据怎么放

训练和验证数据都用 `metadata.jsonl` 驱动。每一行需要包含：

```json
{"original_image":"images/0001_input.jpg","edited_image":"images/0001_target.jpg","edit_prompt":"make it metallic blue with a clean background"}
```

推荐结构：

```text
training/data/custom_train/
  images/
    0001_input.jpg
    0001_target.jpg
    0002_input.jpg
    0002_target.jpg
  metadata.jsonl
```

`original_image` 和 `edited_image` 支持相对路径，也支持 Windows 绝对路径。相对路径默认相对于对应的 `metadata.jsonl` 所在目录解析。

## 2. 快速生成 metadata.jsonl

如果你的图片命名像这样：

```text
chair_01_input.jpg
chair_01_target.jpg
chair_02_input.jpg
chair_02_target.jpg
```

可以直接扫描并生成：

```bash
python training/prepare_metadata.py ^
  --source-dir training/raw/train ^
  --output-metadata training/data/custom_train/metadata.jsonl ^
  --default-prompt "make it metallic blue with a clean background"
```

也支持 CSV/JSON 清单：

```bash
python training/prepare_metadata.py ^
  --manifest-csv training/raw/train_manifest.csv ^
  --output-metadata training/data/custom_train/metadata.jsonl
```

`CSV` 需要至少三列：

```text
original_image,edited_image,edit_prompt
```

脚本会把图片复制到 `metadata.jsonl` 同级目录下的 `images/`，并生成相对路径引用。

## 3. 运行训练

先安装训练依赖，然后直接运行脚本即可。单卡或 CPU 环境都可以用，单卡推荐直接：

```bash
python training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora
```

如果你已经配置了 `accelerate`，也可以用：

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora
```

如果 Hugging Face 数据集本身没有单独的 validation split，可以直接从 train 自动切一部分出来：

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --dataset-name timbrooks/instructpix2pix-clip-filtered ^
  --train-split train ^
  --validation-from-train-ratio 0.05 ^
  --original-image-column original_image ^
  --edited-image-column edited_image ^
  --edit-prompt-column edit_prompt ^
  --output-dir training/outputs/checkpoints/hf_baseline
```

`--val-split` 和 `--validation-from-train-ratio` 二选一，不要同时传。
## 4. 最小训练命令示例

先验证链路：

```bash
python training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora-baseline ^
  --resolution 256 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 5e-5 ^
  --max-train-steps 1000
```

更正式的训练建议：

```bash
python training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora-512 ^
  --resolution 512 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 8 ^
  --learning-rate 1e-4 ^
  --max-train-steps 3000
```

## 5. 推荐起始参数

基线测试：

```text
resolution = 256
train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 5e-5
max_train_steps = 1000
```

正式训练建议：

```text
resolution = 512
train_batch_size = 1
gradient_accumulation_steps = 4~8
learning_rate = 5e-5 或 1e-4
max_train_steps = 2000~5000
```

默认脚本会把这段目标风格后缀追加到每个 prompt：

```text
Keep a single centered object, clean background, clear silhouette, product-style view, suitable for 3D asset generation.
```

如果你不想自动追加，可以在训练和推理时传：

```bash
--prompt-suffix ""
```

## 6. 训练输出

训练结果默认保存在：

```text
training/outputs/checkpoints/pix2pix-lora/
```

其中：

```text
lora/
```

是最终 LoRA 权重目录。

而：

```text
checkpoint-000500/
checkpoint-001000/
```

会保存中间 checkpoint 和对应的 LoRA 权重。

验证生成的对比图会保存在：

```text
training/outputs/samples/<run_name>/
```

每张图从左到右是：

```text
original | generated | target
```

TensorBoard 日志默认保存在：

```text
training/outputs/checkpoints/<run_name>/tensorboard/
```

也可以通过参数自定义日志目录：

```bash
--tensorboard-log-dir training/outputs/tensorboard/custom_run
```

启动方式：

```bash
tensorboard --logdir training/outputs/checkpoints/pix2pix-lora/tensorboard
```

如果你希望一次查看多个实验，可以把上层目录直接传给 TensorBoard：

```bash
tensorboard --logdir training/outputs/checkpoints
```

你会看到这些可视化：

```text
train/loss
train/learning_rate
val/loss
val/previews/sample_00 ...
```

### 6.1 TensorBoard 里应该重点看什么

`train/loss`

训练集上的噪声预测损失。正常情况下，它应该总体向下，允许中间有抖动。

`val/loss`

验证集上的平均损失，用于辅助判断是否真的在泛化，而不是只记住训练样本。脚本会用 `--validation-loss-batches` 指定的 batch 数做近似统计，默认是 `8`。

`train/learning_rate`

当前学习率曲线。它能帮助你确认 scheduler 是否按预期工作，比如常数、线性衰减或者 cosine。

`val/previews/sample_00`

验证样本拼图。每张图从左到右是：

```text
original | generated | target
```

这通常比单看 loss 更重要，因为你的目标是“更适合 TRELLIS 的图像编辑效果”，最终还是要看图像是否真的更干净、更稳、更符合 prompt。

### 6.2 为什么没有 accuracy 曲线

这是扩散图像编辑训练，不是分类或检测任务，所以通常没有有意义的“accuracy 上升曲线”。

对于你的任务，更实用的观察方式是：

```text
loss 是否整体下降
validation preview 是否逐步接近 target
主体是否更稳定
背景是否更干净
prompt 控制是否更明显
```

### 6.3 怎么判断训练在变好

比较理想的现象：

```text
train/loss 缓慢下降
val/loss 大体跟随下降或保持稳定
validation preview 中主体越来越集中、轮廓更清楚
生成图中的背景噪声逐步减少
prompt 指令的风格、材质、颜色控制越来越明显
```

需要警惕的现象：

```text
train/loss 下降但 val/loss 持续上升
validation preview 越来越像训练集 target 的固定模板
生成图开始出现过度锐化、脏背景、重复物体
颜色或材质指令变强了，但主体结构反而更差
```

如果出现这些情况，通常可以尝试：

```text
降低学习率
减少训练步数
增加或清洗训练数据
提高验证频率
检查 prompt 是否太单一
```

### 6.4 建议的观察节奏

先做链路验证时，建议每隔 `100~250` step 看一次 TensorBoard。

正式训练时，建议重点对比：

```text
step 500
step 1000
step 2000
step 3000
```

不要只看最后一个 checkpoint。很多时候最佳 LoRA 会出现在中间阶段。

### 6.5 常用可视化相关参数

```bash
--validation-steps 250
--validation-loss-batches 8
--tensorboard-log-dir training/outputs/checkpoints/pix2pix-lora/tensorboard
```

如果显存或时间比较紧，可以把验证频率调低一些，例如：

```bash
--validation-steps 500
--validation-loss-batches 4
```

### 6.6 一个完整示例

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --train-metadata training/data/custom_train/metadata.jsonl ^
  --val-metadata training/data/custom_val/metadata.jsonl ^
  --output-dir training/outputs/checkpoints/pix2pix-lora ^
  --resolution 256 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 5e-5 ^
  --max-train-steps 1000 ^
  --validation-steps 250 ^
  --validation-loss-batches 8
```

训练启动后，另开一个终端运行：

```bash
tensorboard --logdir training/outputs/checkpoints/pix2pix-lora/tensorboard
```

然后在浏览器里打开 TensorBoard 页面查看曲线和预览图。

## 7. 推理验证

训练结束后，可以快速验证 LoRA 效果：

```bash
python training/infer_lora_pix2pix.py ^
  --lora-path training/outputs/checkpoints/pix2pix-lora/lora ^
  --image assets/example.jpg ^
  --prompt "make it metallic blue with a clean background" ^
  --output outputs/edited/example_lora.png
```

常用可调参数：

```text
--num-inference-steps 20
--guidance-scale 7.5
--image-guidance-scale 1.5
--lora-scale 1.0
```

## 8. 接回当前 demo

当前仓库里的 `pipelines/image_editor.py` 已经做了小扩展，未来可以通过环境变量切换到 LoRA：

```bash
set INSTRUCT_PIX2PIX_LORA_PATH=training\outputs\checkpoints\pix2pix-lora\lora
set INSTRUCT_PIX2PIX_LORA_SCALE=1.0
```

这样不改 TRELLIS 协议，也不需要改 TRELLIS 后端，只是把前端图像编辑器替换成 LoRA 版本。







## 9. TRELLIS 黑盒重排序

训练脚本现在支持在常规 validation 阶段额外跑一轮面向 TRELLIS 的下游评测。

启用 `--enable-trellis-rerank` 后，每次 validation 会执行这条链路：

```text
validation 样本
-> 当前 LoRA 编辑器生成 edited image
-> 把 edited image 送入 TRELLIS
-> TRELLIS 返回 GLB
-> 对 GLB 渲染固定 canonical views
-> 计算 proxy metrics
-> 用 trellis/mean_score 自动选择 best checkpoint
```

### 9.1 会新增保存哪些结果

在你的实验目录下还会新增：

```text
best_checkpoint/
  lora/
  best_checkpoint.json

trellis_eval/
  step_000250/
    sample_00/
      edited_input.png
      generated.png
      original.png
      target.png
      trellis_result.glb
      render_front.png
      render_left.png
      render_right.png
      render_back.png
      render_top.png
      trellis_preview.png
      trellis_metrics.json
    summary.json
```

`best_checkpoint/lora/`

这是当前按照下游 TRELLIS 重排序分数选出来的最佳 LoRA。前端联调时，通常优先加载这个目录。

`best_checkpoint/best_checkpoint.json`

这里会记录最佳 step、所采用的指标、对应的 `val/loss`，以及 TRELLIS 汇总结果。

### 9.2 TensorBoard 新增指标

启用 rerank 后，TensorBoard 还会看到：

```text
trellis/mean_score
trellis/success_rate
trellis/front_similarity
trellis/coverage_score
trellis/centering_score
trellis/view_consistency_score
trellis/previews/sample_00 ...
```

这些是用于模型选择的 proxy metrics，不是可反向传播的训练损失。

### 9.3 推荐训练命令

```bash
accelerate launch training/train_lora_pix2pix.py ^
  --dataset-name timbrooks/instructpix2pix-clip-filtered ^
  --train-split train ^
  --validation-from-train-ratio 0.05 ^
  --original-image-column original_image ^
  --edited-image-column edited_image ^
  --edit-prompt-column edit_prompt ^
  --output-dir training/outputs/checkpoints/hf_pix2pix_lora ^
  --resolution 256 ^
  --train-batch-size 1 ^
  --gradient-accumulation-steps 4 ^
  --learning-rate 5e-5 ^
  --max-train-steps 1000 ^
  --validation-steps 250 ^
  --validation-loss-batches 8 ^
  --enable-trellis-rerank ^
  --trellis-eval-samples 4 ^
  --trellis-render-size 256
```

### 9.4 额外依赖

TRELLIS rerank 渲染阶段除了原本训练依赖外，还需要：

```text
trimesh
pyrender
PyOpenGL
pyglet<2
```

### 9.5 接入前端

训练完成后，你可以直接让前端加载下游最优 checkpoint：

```bash
set INSTRUCT_PIX2PIX_LORA_PATH=training\outputs\checkpoints\hf_pix2pix_lora\best_checkpoint\lora
set INSTRUCT_PIX2PIX_LORA_SCALE=1.0
python app.py
```

如果你想改回最终一步保存的权重，仍然使用 `training/outputs/checkpoints/<run_name>/lora` 即可。
