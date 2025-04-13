# 文本到视频生成服务

这个项目实现了一个文本到视频的生成服务，通过集成CosyVoice语音克隆和EchoMimic人脸动画生成技术，可以将文本转换成带有指定人物形象说话的视频。支持多段文本输入，每段文本会生成一个独立的视频。

## 项目结构

```
proj/
│
├── main_new.py          # FastAPI主程序
├── run_cosyvoice.py     # CosyVoice运行脚本（自动生成）
├── run_echomimic.py     # EchoMimic运行脚本（自动生成）
├── requirements.txt     # 主程序依赖
├── run_service.bat      # 批处理启动文件
│
├── CosyVoice/           # CosyVoice模型目录
│   ├── requirements.txt # CosyVoice依赖
│   ├── cosyvoice_with_azure.py  # CosyVoice模型调用实现
│   └── pretrained_models/       # 预训练模型目录
│
├── Echomimic/           # EchoMimic模型目录
│   ├── .glut/           # EchoMimic本地Python环境
│   └── webgui_glut.py   # EchoMimic模型调用实现
│
├── static/              # 静态文件目录
│   └── outputs/         # 输出文件保存目录
│
├── results/             # 最终视频文件目录
│
└── temp/                # 临时文件目录
```

## 环境要求

### 主程序环境

主程序使用Python 3.8+，并依赖于以下库：
- FastAPI
- Uvicorn
- Python-multipart
- Pydantic
- 其他依赖（详见requirements.txt）

可以通过以下命令安装主程序依赖：
```bash
pip install -r requirements.txt
```

### CosyVoice环境

CosyVoice模型支持两种环境配置方式：

1. 使用本地预配置环境（推荐）：
   - 路径：`./CosyVoice/Cosyvoice1/`
   - 已包含所有必要的依赖和配置
   - 直接使用即可，无需额外配置

2. 自行创建虚拟环境：
   - 创建新的Python虚拟环境
   - 安装必要的依赖：
     - torch
     - torchaudio
     - azure-cognitiveservices-speech
     - 其他依赖（详见CosyVoice/requirements.txt）

### EchoMimic环境

EchoMimic模型使用本地Python环境：
- 路径：`./Echomimic/.glut/python.exe`
- 主要依赖：torch, diffusers, omegaconf, PIL, cv2, facenet-pytorch等

## 启动服务

使用以下命令启动服务：

```bash
python main_new.py
```

或者直接运行批处理文件：

```bash
run_service.bat
```

服务将在http://localhost:9000上运行。

## API说明

### 1. 文本到视频转换

**请求**:
- URL: `/text-to-video/`
- 方法: `POST`
- 表单数据:
  - `texts`: 要转换的文本列表（可以包含多段文本）
  - `prompt_audio`: 提示音频文件（用于语音克隆）
  - `reference_image`: 参考图像文件（用于人脸动画）

**响应**:
```json
{
  "task_id": "uuid-字符串",
  "status": "processing",
  "total_segments": 3
}
```

### 2. 获取任务状态

**请求**:
- URL: `/tasks/{task_id}`
- 方法: `GET`

**响应**:
```json
{
  "task_id": "uuid-字符串",
  "status": "completed",
  "duration": 350.5,
  "video_urls": [
    "/static/outputs/uuid-字符串/video_uuid-字符串_1.mp4",
    "/static/outputs/uuid-字符串/video_uuid-字符串_2.mp4",
    "/static/outputs/uuid-字符串/video_uuid-字符串_3.mp4"
  ],
  "total_segments": 3,
  "progress": 100.0
}
```

状态可能的值：
- `processing`: 处理中
- `generating_audio_1`, `generating_audio_2`...: 正在生成某段文本的音频
- `generating_video_1`, `generating_video_2`...: 正在生成某段文本的视频
- `completed`: 完成
- `failed`: 失败

### 3. 获取单个视频文件

**请求**:
- URL: `/video/{task_id}/{segment}`
- 方法: `GET`
- 参数：
  - `segment`: 段落编号，从1开始

**响应**: 视频文件下载

### 4. 获取所有视频信息

**请求**:
- URL: `/videos/{task_id}`
- 方法: `GET`

**响应**:
```json
{
  "task_id": "uuid-字符串",
  "total_segments": 3,
  "videos": [
    {
      "segment": 1,
      "video_url": "/static/outputs/uuid-字符串/video_uuid-字符串_1.mp4",
      "download_url": "/video/uuid-字符串/1"
    },
    {
      "segment": 2,
      "video_url": "/static/outputs/uuid-字符串/video_uuid-字符串_2.mp4",
      "download_url": "/video/uuid-字符串/2"
    },
    {
      "segment": 3,
      "video_url": "/static/outputs/uuid-字符串/video_uuid-字符串_3.mp4",
      "download_url": "/video/uuid-字符串/3"
    }
  ]
}
```

## 处理流程

1. 接收多段文本和参考音频/图像
2. 对每段文本顺序处理：
   a. 使用CosyVoice模型（cross_lingual模式）生成与参考音频相似声音的语音
   b. 使用EchoMimic模型根据生成的语音和参考图像生成说话视频
3. 返回所有生成的视频文件

## 资源管理

- 处理过程中会按需加载模型
- 每段文本处理完成后释放资源
- 使用线性处理方式，先完成一段文本的全部处理，再处理下一段
- 每个阶段结束后清理GPU内存

## 实现细节

- 主程序使用相对路径，更易于移植
- 启动时自动生成辅助脚本，不需要手动维护
- 两个模型在各自独立的Python环境中运行，避免依赖冲突
- 通过进程间通信传递数据，确保资源隔离
- 提供详细的进度信息，包括当前处理的段落和总进度百分比

## 注意事项

1. 确保CosyVoice和EchoMimic的本地环境路径正确
2. 确保模型的预训练文件已正确安装
3. 上传的提示音频建议使用5-10秒的清晰人声音频
4. 参考图像建议使用正面清晰的人脸图像
5. 处理大型视频或多段文本时可能需要较长时间，请耐心等待
6. 文本段落越多，所需的处理时间和资源也越多 

## Important Note About Repository Contents

Some directories are not included in this repository due to their large size:

1. **Model files**: 
   - `CosyVoice/pretrained_models/`
   - `Echomimic/pretrained_weights/`
   - `.huggingface/`

2. **Environment directories**:
   - `CosyVoice/Cosyvoice1/`
   - `Echomimic/.glut/`

3. **Output and temporary files**:
   - `CosyVoice/outputs/`
   - `results/`
   - `temp/`

## Setting Up the Required Files

To use this project after cloning, you'll need to:

1. **Set up model files**:
   - Download CosyVoice models to `CosyVoice/pretrained_models/`
   - Download EchoMimic weights to `Echomimic/pretrained_weights/`

2. **Create Python environments**:
   - For CosyVoice: Create a Conda environment named "Cosyvoice1"
   - For EchoMimic: Use the provided local environment in `Echomimic/.glut/`

3. **Create directories**:
   - Create directories for `results/` and `temp/` for storing generated files

## Setting Up CosyVoice Environment

CosyVoice requires a Conda environment. Follow these steps to set it up:

1. **Install Miniconda** (if not already installed):
   - Download and install from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)

2. **Create a new Conda environment**:
   ```bash
   conda create -n Cosyvoice1 python=3.10 -y
   ```

3. **Activate the environment**:
   ```bash
   conda activate Cosyvoice1
   ```

4. **Install dependencies**:
   ```bash
   cd CosyVoice
   pip install -r requirements.txt
   ```

5. **Important**: Make sure the environment name is exactly "Cosyvoice1" as this is hardcoded in the startup scripts.

Please refer to the installation instructions in the documentation for more detailed setup guidance.
