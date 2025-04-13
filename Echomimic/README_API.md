# EchoMimic FastAPI 服务

这是 EchoMimic 的 FastAPI 版本，提供 REST API 接口用于生成语音驱动的面部动画视频。

## 功能特点

- 异步处理视频生成请求
- 支持任务状态查询
- 提供视频下载接口
- 与原始 Gradio 界面支持相同的参数配置

## 安装和使用

### 运行服务

Windows 下直接双击 `run_api.bat` 文件启动服务。

### API 端点

服务启动后，访问以下地址查看 API 文档：

```
http://127.0.0.1:7688/docs
```

## API 接口说明

### 1. 生成视频

**端点**: `/generate_video`  
**方法**: `POST`  
**描述**: 上传参考图像和音频文件，生成视频  

**参数**:
- `reference_image`: 参考图像文件（必需）
- `audio_file`: 音频文件（必需）
- `width`: 视频宽度，默认 512
- `height`: 视频高度，默认 512
- `length`: 视频长度，默认 1200
- `seed`: 随机种子，默认 420
- `facemask_dilation_ratio`: 面部蒙版扩张比例，默认 0.1
- `facecrop_dilation_ratio`: 面部裁剪扩张比例，默认 0.5
- `context_frames`: 上下文帧数，默认 12
- `context_overlap`: 上下文重叠帧数，默认 3
- `cfg`: CFG值，默认 1.0
- `steps`: 推理步数，默认 6
- `sample_rate`: 音频采样率，默认 16000
- `fps`: 视频帧率，默认 24
- `device`: 设备类型，默认 "cuda"

**响应示例**:
```json
{
  "task_id": "3f7b53e4-5c4a-4b3d-8b7e-9f8a2d4c6e1b",
  "status": "pending"
}
```

### 2. 获取任务状态

**端点**: `/task/{task_id}`  
**方法**: `GET`  
**描述**: 查询任务处理状态  

**参数**:
- `task_id`: 任务ID（路径参数）

**响应示例**:
```json
{
  "task_id": "3f7b53e4-5c4a-4b3d-8b7e-9f8a2d4c6e1b",
  "status": "completed",
  "output_file": "output/videos/3f7b53e4-5c4a-4b3d-8b7e-9f8a2d4c6e1b_with_audio.mp4",
  "error": null
}
```

### 3. 下载视频

**端点**: `/download/{task_id}`  
**方法**: `GET`  
**描述**: 下载生成的视频文件  

**参数**:
- `task_id`: 任务ID（路径参数）

**响应**: 视频文件下载

## 客户端示例

### Python

```python
import requests
import time
import os

# 服务器URL
API_URL = "http://127.0.0.1:7688"

# 上传文件并生成视频
def generate_video(image_path, audio_path, params=None):
    if params is None:
        params = {}
    
    files = {
        'reference_image': ('image.png', open(image_path, 'rb'), 'image/png'),
        'audio_file': ('audio.wav', open(audio_path, 'rb'), 'audio/wav')
    }
    
    response = requests.post(f"{API_URL}/generate_video", files=files, data=params)
    return response.json()

# 获取任务状态
def get_task_status(task_id):
    response = requests.get(f"{API_URL}/task/{task_id}")
    return response.json()

# 下载生成的视频
def download_video(task_id, save_path):
    response = requests.get(f"{API_URL}/download/{task_id}", stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

# 使用示例
if __name__ == "__main__":
    # 生成视频
    result = generate_video(
        "reference.png", 
        "audio.wav",
        {
            "width": 512,
            "height": 512,
            "steps": 6
        }
    )
    
    task_id = result["task_id"]
    print(f"任务ID: {task_id}")
    
    # 等待任务完成
    while True:
        status = get_task_status(task_id)
        print(f"任务状态: {status['status']}")
        
        if status["status"] in ["completed", "failed"]:
            break
            
        time.sleep(5)  # 每5秒检查一次
    
    # 下载视频
    if status["status"] == "completed":
        print("正在下载视频...")
        if download_video(task_id, "output_video.mp4"):
            print(f"视频下载成功: output_video.mp4")
        else:
            print("视频下载失败")
    else:
        print(f"任务失败: {status.get('error')}")
```

### JavaScript

```javascript
// 使用fetch API的示例
async function generateVideo(imageFile, audioFile, params = {}) {
  const formData = new FormData();
  formData.append('reference_image', imageFile);
  formData.append('audio_file', audioFile);
  
  // 添加其他参数
  Object.keys(params).forEach(key => {
    formData.append(key, params[key]);
  });
  
  const response = await fetch('http://127.0.0.1:7688/generate_video', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

async function getTaskStatus(taskId) {
  const response = await fetch(`http://127.0.0.1:7688/task/${taskId}`);
  return await response.json();
}

// 使用示例
document.getElementById('submitBtn').addEventListener('click', async () => {
  const imageFile = document.getElementById('imageInput').files[0];
  const audioFile = document.getElementById('audioInput').files[0];
  
  if (!imageFile || !audioFile) {
    alert('请选择图片和音频文件');
    return;
  }
  
  const result = await generateVideo(imageFile, audioFile);
  console.log('任务已提交:', result);
  
  const taskId = result.task_id;
  
  // 定期检查任务状态
  const statusCheck = setInterval(async () => {
    const status = await getTaskStatus(taskId);
    console.log('任务状态:', status);
    
    if (status.status === 'completed') {
      clearInterval(statusCheck);
      // 显示下载链接
      const downloadLink = document.createElement('a');
      downloadLink.href = `http://127.0.0.1:7688/download/${taskId}`;
      downloadLink.textContent = '下载生成的视频';
      downloadLink.download = `echomimic_${taskId}.mp4`;
      document.body.appendChild(downloadLink);
    } else if (status.status === 'failed') {
      clearInterval(statusCheck);
      console.error('任务失败:', status.error);
    }
  }, 5000);
});
``` 