# External Experts Module

External Experts 模块包含了专门用于空间智能任务的专业模型，包括深度估计、目标检测、分割、3D重建等功能。所有工具都采用 server/client 架构，支持独立部署和调用。

## 📁 模块结构

```
external_experts/
├── __init__.py                     # 模块初始化
├── checkpoints/                    # 所有模型权重文件
│   └──depth_anything
│   └──grounding_dino
│   └──pi3
│   └──pi3x
│   └──sam2
├── GroundingDINO/                  # 开放词汇目标检测
├── SAM2/                          # 图像和视频分割
├── Depth_AnythingV2/              # 深度估计
├── Pi3/                           # 3D重建 (Pi3 & Pi3X)
├── moondream/                     # 视觉语言模型
└── supervision/                   # YOLO目标检测和标注工具
```

## 🛠️ 工具概览

| 工具名称 | Tool Class | 功能 | 主要用途 | 默认端口 | 主要参数 |
|---------|------------|------|----------|----------|----------|
| **Depth AnythingV2** | `DepthEstimationTool` | 深度估计 | 单目深度估计，分析图像中的3D深度关系 | 20019 | `image_path` |
| **SAM2** | `SegmentationTool` | 图像/视频分割 | 高精度分割任务，精确分割图像中的对象 | 20020 | `image_path`, `point_coords`(可选), `point_labels`(可选), `box`(可选) |
| **GroundingDINO** | `ObjectDetectionTool` | 开放词汇目标检测 | 基于文本描述检测任意物体 | 20022 | `image_path`, `text_prompt`, `box_threshold`, `text_threshold` |
| **Moondream** | `MoondreamTool` | 视觉语言模型 | 图像理解和问答，基于图像内容回答自然语言问题 | 20024 | `image_path`, `task`, `object_name` |
| **Pi3** | `Pi3Tool` | 3D重建 | 从图像生成3D点云和多视角渲染图 | 20030 | `image_path`, `azimuth_angle`, `elevation_angle` |
| **Pi3X** | `Pi3XTool` | 3D重建（增强版） | Pi3升级版，更平滑点云、近似度量尺度、可选多模态条件注入 | 20031 | `image_path`, `azimuth_angle`, `elevation_angle` |
| **Supervision** | `SupervisionTool` | 目标检测标注 | YOLO模型和可视化工具，通用目标检测和分割 | - | `image_path`, `task` ("image_det" 或 "image_seg") |
| **YOLO-E** | `YOLOETool` | YOLO-E检测 | 高精度检测，支持自定义类别 | - | `image_path`, `task`, `class_names` |

**使用示例**:
- 详细使用示例请参考：[Advanced Examples](../Examples/ADVANCED_EXAMPLES.md)
- 快速入门指南请参考：[Quick Start Guide](../../readme.md#-quick-start)

---

## 📋 详细工具介绍

### 1. Depth AnythingV2 - 深度估计

**功能**: 单目图像深度估计

**特点**:
- 三种模型规格可选
- 高质量深度图生成
- 支持多种输入格式

**文件结构**:
```
Depth_AnythingV2/
├── depth_server.py
├── depth_client.py
├── mock_depth_service.py
└── depth_anything_v2/
```

**模型规格**:
| 模型 | 骨干网络 | 参数量 | 文件大小 | 推理速度 | 精度 |
|------|----------|--------|----------|----------|------|
| Small | ViT-S | ~25M | ~100MB | 快 | 良好 |
| Base | ViT-B | ~97M | ~390MB | 中等 | 高 |
| Large | ViT-L | ~335M | ~1.3GB | 慢 | 很高 |

**权重下载**:
```bash
cd checkpoints/
# Small模型 (~25MB, 最快)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
# Base模型 (~100MB, 平衡) - 推荐
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
# Large模型 (~350MB, 最高质量)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

**资源链接**:
- [官方仓库](https://github.com/DepthAnything/Depth-Anything-V2)
- [论文](https://arxiv.org/abs/2406.09414)

---


### 2. SAM2 - 图像和视频分割

**功能**: 高精度的图像和视频分割模型

**特点**:
- 支持图像和视频分割
- 多种模型规格可选
- 高精度分割效果

**文件结构**:
```
SAM2/
├── sam2_server.py
└── sam2_client.py
```

**模型规格**:
| 模型 | 参数量 | 文件大小 | 用途 |
|------|--------|----------|------|
| Hiera Large | ~224M | ~900MB | 高精度 |
| Hiera Base+ | ~80M | ~320MB | 平衡性能 |
| Hiera Small | ~46M | ~185MB | 快速推理 |

**权重下载**:
#### 使用官方脚本（推荐）
```bash
cd checkpoints/
# 推荐使用官方脚本
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```

#### 手动下载
```bash
cd checkpoints/

# SAM2.1 Hiera Large (推荐)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# SAM2.1 Hiera Base+ 
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# SAM2.1 Hiera Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

**资源链接**:
- [官方仓库](https://github.com/facebookresearch/sam2)
- [论文](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)

---

### 3. GroundingDINO - 开放词汇目标检测

**功能**: 基于自然语言描述检测图像中的目标物体

**特点**:
- 支持开放词汇检测，无需预定义类别
- 基于Swin-B骨干网络
- 可通过文本描述检测任意物体

**文件结构**:
```
GroundingDINO/
├── grounding_dino_server.py
├── grounding_dino_client.py
└── configs/
    └── GroundingDINO_SwinB_cfg.py
```

**安装**:
```bash
pip install groundingdino_py
```

**权重下载**:
```bash
cd checkpoints/
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

**资源链接**:
- [官方仓库](https://github.com/IDEA-Research/GroundingDINO)
- [论文](https://arxiv.org/abs/2303.05499)

---

### 4. Moondream - 视觉语言模型

**功能**: 视觉语言理解和图像问答

**特点**:
- 图像理解能力
- 自然语言交互
- API接口支持

**文件结构**:
```
moondream/
├── md_server.py          # 服务器端
├── md_client.py          # 客户端
├── md_local.py          # 本地部署
├── __init__.py
└── __pycache__/
```

**安装**:
```bash
pip install moondream
```

**环境配置**:
```bash
export MOONDREAM_API_KEY="your_api_key"
```

**资源链接**:
- [官方网站](https://moondream.ai/)
- [API文档](https://docs.moondream.ai/)

---

### 5. Pi3 - 3D重建服务

**功能**: 基于Pi3模型的3D重建，从图像生成3D点云

**特点**:
- 高质量3D重建
- 支持PLY格式输出
- 可视化支持

**文件结构**:
```
Pi3/
├── pi3/                  # 运行代码
├── example.py            # 原始Pi3运行代码
├── pi3_server.py         # Flask服务器
└── pi3_client.py         # 客户端
```

**环境要求**:
- torch==2.5.1
- torchvision==0.20.1
- numpy==1.26.4

**使用方法**:
```bash
# 可视化生成的PLY文件
python spagent/utils/ply_to_html_viewer.py xxx.ply --output xxx.html --max_points 100000
```

**权重下载**:
```bash
cd checkpoints/pi3
wget https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```

---

### 5.1 Pi3X - 增强版3D重建服务

**功能**: 基于Pi3X模型的增强版3D重建（Pi3的升级版本）

**特点**:
- 更平滑的点云重建（ConvHead替代LinearPts3d，消除网格伪影）
- 近似度量尺度重建（Metric Scale）
- 可选多模态条件注入（相机位姿、内参、深度）
- 更可靠的连续置信度评分
- 与Pi3完全兼容的API接口（相同的输入/输出格式）

**文件结构**:
```
Pi3/
├── pi3/
│   └── models/
│       ├── pi3.py            # 原始Pi3模型
│       ├── pi3x.py           # Pi3X模型（增强版）
│       └── layers/
│           ├── conv_head.py   # 卷积上采样Head（Pi3X）
│           └── prope.py       # PRoPE位置编码（Pi3X）
├── pi3_server.py              # Pi3 Flask服务器
├── pi3_client.py              # Pi3客户端
├── pi3x_server.py             # Pi3X Flask服务器
└── pi3x_client.py             # Pi3X客户端
```

**权重下载**:
```bash
mkdir -p checkpoints/pi3x
cd checkpoints/pi3x
wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors
```

**资源链接**:
- [官方仓库](https://github.com/yyfz/Pi3)
- [Pi3X HuggingFace权重](https://huggingface.co/yyfz233/Pi3X)

---

### 6. Supervision - 目标检测和标注工具

**功能**: YOLO目标检测和可视化标注工具

**特点**:
- 集成多种YOLO模型
- 丰富的可视化工具
- 标注和后处理功能

**文件结构**:
```
supervision/
├── __init__.py
├── supervision_server.py
├── supervision_client.py
├── sv_yoloe_server.py
├── sv_yoloe_client.py
├── annotator.py
├── yoloe_annotator.py
├── yoloe_test.py 
├── download_weights.py
└── mock_supervision_service.py
```

**安装**:
```bash
pip install supervision
```

**可用模型**:
| 模型文件 | 功能 | 用途 |
|----------|------|------|
| yoloe-v8l-seg.pt | YOLOE v8 Large 分割 | 高精度目标检测和分割 |
| yoloe-v8l-seg-pf.pt | YOLOE v8 Large 分割(优化版) | 性能优化的分割模型 |

**权重下载**:
```bash
python download_weights.py
```

**资源链接**:
- [官方仓库](https://github.com/roboflow/supervision)
- [文档](https://supervision.roboflow.com/)

---

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
# 需要GPU内存 >= 24G
apt-get install tmux
pip install torch torchvision
pip install groundingdino_py supervision moondream
```

创建checkpoints目录：
```bash
mkdir -p checkpoints/{grounding_dino,depth_anything,pi3,pi3x,sam2}
```
### 2. 下载模型权重

每个工具都需要下载相应的模型权重文件，请参考各工具的详细说明。

### 3. 启动服务

如果要使用真实的专家服务而非mock模式，根据需要启动相应的服务器：
```bash
# 深度估计服务
python spagent/external_experts/Depth_AnythingV2/depth_server.py \
  --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth \
  --port 20019

# 部署SAM2分割服务，这里面需要将sam的权重名字rename成sam2.1_b.pt，否则会报错
python spagent/external_experts/SAM2/sam2_server.py \
  --checkpoint_path checkpoints/sam2/sam2.1_b.pt \
  --port 20020

# 部署grounding dino
# sometimes the network cannot connect the huggingface, we can reset the huggingfacesource
export HF_ENDPOINT=https://hf-mirror.com

python spagent/external_experts/GroundingDINO/grounding_dino_server.py \
  --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth \
  --port 20022

# 3D重建服务（Pi3）
python spagent/external_experts/Pi3/pi3_server.py \
  --checkpoint_path checkpoints/pi3/model.safetensors \
  --port 20030

# 3D重建服务（Pi3X - 增强版，推荐）
python spagent/external_experts/Pi3/pi3x_server.py \
  --checkpoint_path checkpoints/pi3x/model.safetensors \
  --port 20031

# 视觉语言模型服务
python spagent/external_experts/moondream/md_server.py \
  --port 20024
```