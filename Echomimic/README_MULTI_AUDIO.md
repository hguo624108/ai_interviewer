# EchoMimic 多段音频API使用说明

## 简介

EchoMimic 是一个能够根据参考图像和音频生成逼真面部表情动画的系统。新版API增加了对多段音频生成多段视频的支持，并可选择性地将这些视频合并为一个完整的视频。

## API端点

除了原有的API端点外，新版本添加了以下端点：

### 1. 生成多段视频

```
POST /generate_multi_videos
```

#### 请求参数

| 参数名 | 类型 | 必填 | 描述 |
|-------|------|------|------|
| reference_image | File | 是 | 参考图像文件（面部照片） |
| audio_files | File[] | 是 | 多个音频文件（WAV格式） |
| width | int | 否 | 视频宽度 (默认: 512) |
| height | int | 否 | 视频高度 (默认: 512) |
| length | int | 否 | 最大长度 (默认: 1200) |
| seed | int | 否 | 随机种子 (默认: 420) |
| facemask_dilation_ratio | float | 否 | 面部蒙版扩展比例 (默认: 0.1) |
| facecrop_dilation_ratio | float | 否 | 面部裁剪扩展比例 (默认: 0.5) |
| context_frames | int | 否 | 上下文帧数 (默认: 12) |
| context_overlap | int | 否 | 上下文重叠帧数 (默认: 3) |
| cfg | float | 否 | 配置参数 (默认: 1.0) |
| steps | int | 否 | 生成步骤数 (默认: 6) |
| sample_rate | int | 否 | 音频采样率 (默认: 16000) |
| fps | int | 否 | 视频帧率 (默认: 24) |
| device | string | 否 | 计算设备 (默认: "cuda") |
| combine_videos | bool | 否 | 是否合并视频 (默认: false) |

#### 响应

```json
{
  "task_id": "任务ID字符串",
  "status": "pending"
}
```

### 2. 获取任务状态

```
GET /task/{task_id}
```

#### 响应

```json
{
  "task_id": "任务ID字符串",
  "status": "pending|processing|completed|failed",
  "output_files": ["视频片段1路径", "视频片段2路径", ...],
  "combined_output_file": "合并视频路径",
  "error": "失败原因（如果有）"
}
```

### 3. 下载视频片段

```
GET /download_segment/{task_id}/{segment_id}
```

返回生成的视频片段文件。

### 4. 下载合并的视频

```
GET /download_combined/{task_id}
```

返回合并后的完整视频文件（如果设置了`combine_videos = true`）。

## 使用示例

### Python 示例代码

```python
import requests

# 服务器地址
url = "http://localhost:7688/generate_multi_videos"

# 准备文件
files = []
files.append(
    ('reference_image', ('face.png', open('face.png', 'rb'), 'image/png'))
)

# 添加多个音频文件
files.append(
    ('audio_files', ('audio1.wav', open('audio1.wav', 'rb'), 'audio/wav'))
)
files.append(
    ('audio_files', ('audio2.wav', open('audio2.wav', 'rb'), 'audio/wav'))
)

# 其他参数
data = {
    'combine_videos': 'true'  # 将生成的视频片段合并
}

# 发送请求
response = requests.post(url, files=files, data=data)

# 关闭文件句柄
for _, file_tuple in files:
    if hasattr(file_tuple[1], 'close'):
        file_tuple[1].close()

# 检查响应
if response.status_code == 200:
    task_id = response.json()['task_id']
    print(f"任务ID: {task_id}")
else:
    print(f"请求失败: {response.status_code}")
```

### 命令行测试

项目提供了两个测试脚本：

1. `test_multi_audio_api.py`: 完整的测试脚本，支持命令行参数
   ```
   python test_multi_audio_api.py --reference face.png --audio audio1.wav audio2.wav --combine
   ```

2. `test_simple.py`: 简单测试脚本，修改脚本中的文件路径来测试
   ```
   python test_simple.py
   ```

## 注意事项

1. 所有音频文件应为WAV格式
2. 参考图像应包含清晰的人脸
3. 生成大量视频可能需要较长时间
4. 视频生成过程需要较大的GPU内存

## 常见问题

1. **Q: 为什么任务状态一直是processing？**
   A: 视频生成是计算密集型任务，可能需要几分钟到几十分钟不等。

2. **Q: 如何查看任务的详细状态？**
   A: 访问 `GET /task/{task_id}` 接口可获取详细状态。

3. **Q: 服务器返回内存错误怎么办？**
   A: 可能是GPU内存不足，尝试减小视频分辨率或长度。 