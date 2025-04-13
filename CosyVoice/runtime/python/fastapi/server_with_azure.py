import os
import sys
import time
import json
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response
import tempfile
from typing import Optional, List, Dict

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取根目录路径
root_dir = os.path.abspath(os.path.join(current_dir, "../../../.."))
# 设置相对路径
matcha_tts_path = os.path.join(root_dir, "CosyVoice", "third_party", "Matcha-TTS")
cosyvoice_path = os.path.join(root_dir, "CosyVoice")

# 检查路径是否已经在PYTHONPATH中
if matcha_tts_path not in sys.path:
    sys.path.insert(0, matcha_tts_path)
if cosyvoice_path not in sys.path:
    sys.path.insert(0, cosyvoice_path)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice_with_azure import CosyVoiceWithAzure

# 配置路径
DEFAULT_OUTPUTS_DIR = os.path.join(root_dir, "static", "outputs", "audio")
# 确保输出目录存在
os.makedirs(DEFAULT_OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="CosyVoice API")

# 初始化CosyVoice实例
cosyvoice = CosyVoiceWithAzure(
    tts_model_path=os.path.join(root_dir, "CosyVoice", "pretrained_models", "CosyVoice2-0.5B"),
    output_dir=DEFAULT_OUTPUTS_DIR
)

@app.post("/cross_lingual_batch")
async def cross_lingual_batch(
    texts: str = Form(...),  # JSON格式的文本数组或以换行符分隔的多行文本
    prompt_audio: UploadFile = File(...),
    output_dir: Optional[str] = Form(None),
    output_filenames: Optional[str] = Form(None),  # 可选的自定义文件名，JSON数组或换行符分隔
    return_type: str = Form("path"),  # 批量处理默认返回路径
    input_format: str = Form("auto")  # "json", "text" 或 "auto"
):
    """
    批量生成多段语音，也可以只生成一段
    
    参数:
    - texts: 可以是JSON格式的文本数组，如["问题1", "问题2", "问题3"]，
             也可以是以换行符分隔的多行文本，如"问题1\n问题2\n问题3"，
             或者只是单个文本字符串
    - prompt_audio: 上传的音频文件，用作音色参考
    - output_dir: 可选，指定输出文件的保存目录，如不提供则使用默认目录
    - output_filenames: 可选，自定义输出文件名(不含扩展名)，格式与texts一致
    - return_type: 返回类型，"path"返回文件路径，"data"直接返回音频数据
    - input_format: 输入格式，"json"表示JSON格式，"text"表示文本格式，"auto"自动检测
    """
    temp_dir = None
    temp_file = None
    
    try:
        # 解析文本
        text_list = []
        
        # 根据input_format确定解析方式
        if input_format == "json" or (input_format == "auto" and texts.strip().startswith("[")):
            try:
                text_list = json.loads(texts)
                if not isinstance(text_list, list):
                    text_list = [texts]  # 如果是单个文本的JSON格式
            except json.JSONDecodeError:
                # 如果不是有效的JSON，作为单个文本处理
                text_list = [texts]
        elif input_format == "text" or input_format == "auto":
            # 按换行符分割文本，如果没有换行符则视为单个文本
            lines = [line.strip() for line in texts.split("\n") if line.strip()]
            if lines:
                text_list = lines
            else:
                text_list = [texts]
        
        # 验证文本数量
        if len(text_list) == 0:
            raise HTTPException(status_code=400, detail="未提供有效的文本内容")
        if len(text_list) > 10:
            raise HTTPException(status_code=400, detail="最多支持10个文本")
        
        # 解析自定义文件名
        filename_list = []
        if output_filenames:
            if output_filenames.strip().startswith("["):
                try:
                    filename_list = json.loads(output_filenames)
                    if not isinstance(filename_list, list):
                        filename_list = []
                except json.JSONDecodeError:
                    filename_list = []
            else:
                filename_list = [name.strip() for name in output_filenames.split("\n") if name.strip()]
            
            # 确保文件名列表长度与文本列表一致
            if filename_list and len(filename_list) != len(text_list):
                raise HTTPException(status_code=400, detail="自定义文件名数量必须与文本数量一致")
        
        # 保存上传的音频文件到临时文件
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "prompt_audio.wav")
        
        with open(temp_file, "wb") as f:
            f.write(await prompt_audio.read())
        
        # 生成会话ID，用于关联一批文件
        session_id = f"batch_{int(time.time())}"
        
        # 确定输出目录
        if output_dir:
            # 如果提供了绝对路径，直接使用
            if os.path.isabs(output_dir):
                final_output_dir = output_dir
            else:
                # 否则视为相对于根目录的路径
                final_output_dir = os.path.join(root_dir, output_dir)
            # 确保目录存在
            os.makedirs(final_output_dir, exist_ok=True)
        else:
            final_output_dir = DEFAULT_OUTPUTS_DIR
        
        results = []
        
        # 处理每个文本
        for i, text in enumerate(text_list):
            try:
                # 确定输出文件名
                if i < len(filename_list) and filename_list[i]:
                    output_basename = f"{filename_list[i]}"
                else:
                    output_basename = f"{session_id}_{i}"
                
                output_filename = f"{output_basename}.wav"
                output_path = os.path.join(final_output_dir, output_filename)
                
                # 使用临时文件路径作为提示音频路径
                output_paths = cosyvoice.cross_lingual_with_auto_prompt(
                    text,
                    temp_file
                )
                
                if output_paths and len(output_paths) > 0:
                    # 移动到最终路径
                    shutil.copy(output_paths[0], output_path)
                    
                    # 添加结果
                    results.append({
                        "text": text,
                        "index": i,
                        "status": "success",
                        "output_path": output_path,
                        "filename": output_filename
                    })
                else:
                    results.append({
                        "text": text,
                        "index": i,
                        "status": "error",
                        "message": "音频生成失败"
                    })
            except Exception as e:
                results.append({
                    "text": text,
                    "index": i,
                    "status": "error",
                    "message": str(e)
                })
        
        # 如果选择返回数据而不是路径
        if return_type == "data":
            # 读取所有音频数据
            data_results = []
            for result in results:
                if result["status"] == "success":
                    with open(result["output_path"], "rb") as f:
                        import base64
                        audio_base64 = base64.b64encode(f.read()).decode("utf-8")
                        result["audio_data"] = audio_base64
                        # 保留路径信息，但添加音频数据
                    data_results.append(result)
                else:
                    data_results.append(result)
            
            results = data_results
        
        # 返回所有处理结果
        response = {
            "status": "success",
            "session_id": session_id,
            "count": len(text_list),
            "results": results
        }
        
        return response
    
    except Exception as e:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {str(cleanup_error)}")
        
        # 返回错误信息
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)