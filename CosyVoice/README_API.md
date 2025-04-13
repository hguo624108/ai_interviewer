# CosyVoice API 使用指南

## 简介

CosyVoice API 是一个基于FastAPI的语音合成服务，提供了跨语言(cross-lingual)的语音克隆功能。通过提供参考音频和文本，API可以生成与参考音色相似的语音。

## 安装与启动

### 前置条件
- Python 3.7+
- 已安装CosyVoice及其依赖
- 预训练的CosyVoice2-0.5B模型

### 启动服务
以下两种方式均可启动服务：

#### 方式1：使用本地预配置环境（推荐）
```bash
cd CosyVoice/runtime/python/fastapi
..\..\..\Cosyvoice1\python.exe server_with_azure.py
```

#### 方式2：使用自定义虚拟环境
```bash
cd CosyVoice/runtime/python/fastapi
conda activate Cosyvoice1
python server_with_azure.py
```

服务将在 http://localhost:8000 上启动。

## API说明

### `/cross_lingual_batch` 接口

这是主要的语音生成接口，支持单条或多条文本的批量处理。

#### 请求参数

| 参数名 | 类型 | 必填 | 描述 |
|-------|-----|------|------|
| `texts` | 表单字段(Form) | 是 | 文本内容，支持三种格式:<br>1. 单条文本字符串<br>2. 多行文本(用换行符分隔)<br>3. JSON数组 |
| `prompt_audio` | 文件(File) | 是 | 音色参考的音频文件(WAV格式) |
| `output_dir` | 表单字段(Form) | 否 | 指定输出目录，支持绝对路径或相对路径 |
| `output_filenames` | 表单字段(Form) | 否 | 自定义输出文件名(不含扩展名)，格式与texts一致 |
| `return_type` | 表单字段(Form) | 否 | 返回类型，"path"(默认)或"data" |
| `input_format` | 表单字段(Form) | 否 | 输入格式，"auto"(默认)、"json"或"text" |

#### 路径设置说明

- 默认输出目录：`[项目根目录]/static/outputs/audio`
- 如果提供绝对路径作为`output_dir`，将直接使用该路径
- 如果提供相对路径，将相对于项目根目录解析
- 系统会自动创建不存在的目录

#### 文件名设置说明

- 默认文件名格式：`batch_[时间戳]_[索引].wav`
- 如果提供`output_filenames`，将使用自定义文件名
- 文件名不需要包含扩展名，系统会自动添加`.wav`
- 自定义文件名的数量必须与文本数量一致

## 使用示例

### Python示例代码

#### 1. 处理单条文本
```python
import requests

url = "http://localhost:8000/cross_lingual_batch"

# 单条文本
text = "这是一个测试文本"

files = {
    'prompt_audio': ('voice.wav', open('voice.wav', 'rb'), 'audio/wav')
}
data = {
    'texts': text,
    'return_type': 'path'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### 2. 处理多条文本(多行文本格式)
```python
import requests

url = "http://localhost:8000/cross_lingual_batch"

# 多行文本
texts = """第一个问题
第二个问题
第三个问题"""

# 自定义文件名
filenames = """question1
question2
question3"""

files = {
    'prompt_audio': ('voice.wav', open('voice.wav', 'rb'), 'audio/wav')
}
data = {
    'texts': texts,
    'output_filenames': filenames,
    'output_dir': 'my_outputs',  # 相对路径
    'return_type': 'path',
    'input_format': 'text'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### 3. 处理多条文本(JSON格式)
```python
import requests
import json

url = "http://localhost:8000/cross_lingual_batch"

# JSON格式文本
texts = ["问题1", "问题2", "问题3"]
filenames = ["file1", "file2", "file3"]

files = {
    'prompt_audio': ('voice.wav', open('voice.wav', 'rb'), 'audio/wav')
}
data = {
    'texts': json.dumps(texts),
    'output_filenames': json.dumps(filenames),
    'output_dir': 'D:/my_project/outputs',  # 绝对路径
    'return_type': 'data',
    'input_format': 'json'
}

response = requests.post(url, files=files, data=data)
result = response.json()
```

### 返回格式示例

```json
{
  "status": "success",
  "session_id": "batch_1654321098",
  "count": 3,
  "results": [
    {
      "text": "第一个问题",
      "index": 0,
      "status": "success",
      "output_path": "/path/to/question1.wav",
      "filename": "question1.wav"
    },
    {
      "text": "第二个问题",
      "index": 1,
      "status": "success",
      "output_path": "/path/to/question2.wav",
      "filename": "question2.wav"
    },
    {
      "text": "第三个问题",
      "index": 2,
      "status": "success",
      "output_path": "/path/to/question3.wav",
      "filename": "question3.wav"
    }
  ]
}
```

当`return_type="data"`时，每个成功结果中还会包含`audio_data`字段，值为Base64编码的音频数据。

## 注意事项

1. 音频文件格式
   - 参考音频必须是WAV格式
   - 生成的音频也是WAV格式，采样率为16kHz

2. 处理限制
   - 最多支持10条文本同时处理
   - 文本过长可能导致处理时间增加

3. 错误处理
   - 单条文本处理失败不会影响其他文本的处理
   - 请检查响应中每个结果的status字段

4. 资源管理
   - 所有临时文件会在处理完成后自动清理
   - 对于批量处理，建议使用`return_type="path"`以减少数据传输

5. 微服务集成
   - 此API设计为微服务架构的一部分，专注于音频生成
   - 可以与其他服务(如Echomimic)集成，实现更复杂的功能

## 故障排除

1. 如果遇到"无效的JSON格式"错误，检查文本是否符合JSON格式或改用text格式
2. 如果音频生成失败，确保参考音频质量良好且格式正确
3. 如果路径相关错误，检查目录权限和路径是否有效

## 服务配置

服务配置位于`server_with_azure.py`文件中，主要参数包括：

- 模型路径: `tts_model_path` (默认为 `pretrained_models/CosyVoice2-0.5B`)
- 默认输出目录: `DEFAULT_OUTPUTS_DIR` (默认为 `static/outputs/audio`)
- 服务端口: 8000 (可在启动脚本中修改)

如需修改这些配置，请编辑`server_with_azure.py`文件。 