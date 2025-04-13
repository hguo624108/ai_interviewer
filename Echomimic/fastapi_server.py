#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
EchoMimic FastAPI 服务
'''

import os
import random
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from facenet_pytorch import MTCNN
import argparse

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# 默认参数
default_values = {
    "width": 512,
    "height": 512,
    "length": 1200,
    "seed": 420,
    "facemask_dilation_ratio": 0.1,
    "facecrop_dilation_ratio": 0.5,
    "context_frames": 12,
    "context_overlap": 3,
    "cfg": 1.0,
    "steps": 6,
    "sample_rate": 16000,
    "fps": 24,
    "device": "cuda"
}

# 检查和配置FFMPEG路径
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("请下载ffmpeg-static并设置到FFMPEG_PATH。\n例如: export FFMPEG_PATH=/path/to/ffmpeg")
elif ffmpeg_path not in os.getenv('PATH'):
    print("添加ffmpeg到PATH")
    os.environ["PATH"] = f"{ffmpeg_path};{os.environ['PATH']}"

# 加载配置
config_path = "./configs/prompts/animation_acc.yaml"
config = OmegaConf.load(config_path)
if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

inference_config_path = config.inference_config
infer_config = OmegaConf.load(inference_config_path)

print("正在初始化模型...")
############# model_init started #############
## vae init
vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to("cuda", dtype=weight_dtype)

## reference net init
reference_unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_base_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device=device)
reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))

## denoising net init
if os.path.exists(config.motion_module_path):
    ### stage1 + stage2
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
else:
    ### only stage1
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
        }
    ).to(dtype=weight_dtype, device=device)

denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

## face locator init
face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype, device="cuda")
face_locator.load_state_dict(torch.load(config.face_locator_path))

## load audio processor params
audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

## load face detector params
face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)

############# model_init finished #############

sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
scheduler = DDIMScheduler(**sched_kwargs)

pipe = Audio2VideoPipeline(
    vae=vae,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    audio_guider=audio_processor,
    face_locator=face_locator,
    scheduler=scheduler,
).to("cuda", dtype=weight_dtype)

print("模型初始化完成！")

# 创建输出目录
output_dir = Path("output/videos")
output_dir.mkdir(exist_ok=True, parents=True)
temp_dir = Path("output/temp")
temp_dir.mkdir(exist_ok=True, parents=True)

# 创建FastAPI应用
app = FastAPI(
    title="EchoMimic API", 
    description="EchoMimic API - 生成语音驱动的面部动画视频",
    version="1.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
app.mount("/output", StaticFiles(directory="output"), name="output")

# 存储任务状态和结果
tasks = {}

class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    output_file: Optional[str] = None
    output_files: Optional[List[str]] = None
    combined_output_file: Optional[str] = None
    error: Optional[str] = None

def select_face(det_bboxes, probs):
    """选择最大的面部区域"""
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None
    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]

async def process_video_task(
    task_id: str,
    image_path: str,
    audio_path: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    facemask_dilation_ratio: float,
    facecrop_dilation_ratio: float,
    context_frames: int,
    context_overlap: int,
    cfg: float,
    steps: int,
    sample_rate: int,
    fps: int,
    device_type: str
):
    """异步处理视频生成任务"""
    try:
        # 更新任务状态为处理中
        tasks[task_id]["status"] = "processing"
        
        # 设置随机种子
        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(random.randint(100, 1000000))

        # 准备面部蒙版
        face_img = cv2.imread(image_path)
        face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
        det_bboxes, probs = face_detector.detect(face_img)
        select_bbox = select_face(det_bboxes, probs)
        
        if select_bbox is None:
            face_mask[:, :] = 255
        else:
            xyxy = select_bbox[:4]
            xyxy = np.round(xyxy).astype('int')
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            r_pad = int((re - rb) * facemask_dilation_ratio)
            c_pad = int((ce - cb) * facemask_dilation_ratio)
            face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255
            
            # 面部裁剪
            r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
            c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
            crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
            face_img = crop_and_pad(face_img, crop_rect)[0]
            face_mask = crop_and_pad(face_mask, crop_rect)[0]
            face_img = cv2.resize(face_img, (width, height))
            face_mask = cv2.resize(face_mask, (width, height))

        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
        face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
        
        # 生成视频
        video = pipe(
            ref_image_pil,
            audio_path,
            face_mask_tensor,
            width,
            height,
            length,
            steps,
            cfg,
            generator=generator,
            audio_sample_rate=sample_rate,
            context_frames=context_frames,
            fps=fps,
            context_overlap=context_overlap
        ).videos

        # 保存视频
        output_video_path = output_dir / f"{task_id}.mp4"
        save_videos_grid(video, str(output_video_path), n_rows=1, fps=fps)

        # 添加音频
        video_clip = VideoFileClip(str(output_video_path))
        audio_clip = AudioFileClip(audio_path)
        final_output_path = output_dir / f"{task_id}_with_audio.mp4"
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac")
        
        # 清理临时文件
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # 更新任务状态为完成
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["output_file"] = str(final_output_path)
        
        # 释放显存
        torch.cuda.empty_cache()
        
    except Exception as e:
        # 发生错误，更新任务状态
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"任务 {task_id} 处理失败: {e}")
        torch.cuda.empty_cache()

async def process_multi_audio_video_task(
    task_id: str,
    image_path: str,
    audio_paths: List[str],
    width: int,
    height: int,
    length: int,
    seed: int,
    facemask_dilation_ratio: float,
    facecrop_dilation_ratio: float,
    context_frames: int,
    context_overlap: int,
    cfg: float,
    steps: int,
    sample_rate: int,
    fps: int,
    device_type: str,
    combine_videos: bool = False
):
    """异步处理多段音频生成多段视频任务"""
    try:
        # 更新任务状态为处理中
        tasks[task_id]["status"] = "processing"
        
        # 准备面部蒙版
        face_img = cv2.imread(image_path)
        face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
        det_bboxes, probs = face_detector.detect(face_img)
        select_bbox = select_face(det_bboxes, probs)
        
        if select_bbox is None:
            face_mask[:, :] = 255
        else:
            xyxy = select_bbox[:4]
            xyxy = np.round(xyxy).astype('int')
            rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
            r_pad = int((re - rb) * facemask_dilation_ratio)
            c_pad = int((ce - cb) * facemask_dilation_ratio)
            face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255
            
            # 面部裁剪
            r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
            c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
            crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]), min(re + r_pad_crop, face_img.shape[0])]
            face_img = crop_and_pad(face_img, crop_rect)[0]
            face_mask = crop_and_pad(face_mask, crop_rect)[0]
            face_img = cv2.resize(face_img, (width, height))
            face_mask = cv2.resize(face_mask, (width, height))

        ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
        face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(0).unsqueeze(0) / 255.0
        
        output_files = []
        video_clips = []
        
        # 对每个音频片段处理
        for idx, audio_path in enumerate(audio_paths):
            # 设置随机种子
            if seed is not None and seed > -1:
                generator = torch.manual_seed(seed + idx)
            else:
                generator = torch.manual_seed(random.randint(100, 1000000))
            
            # 生成视频
            video = pipe(
                ref_image_pil,
                audio_path,
                face_mask_tensor,
                width,
                height,
                length,
                steps,
                cfg,
                generator=generator,
                audio_sample_rate=sample_rate,
                context_frames=context_frames,
                fps=fps,
                context_overlap=context_overlap
            ).videos

            # 保存视频
            output_video_path = output_dir / f"{task_id}_segment_{idx}.mp4"
            save_videos_grid(video, str(output_video_path), n_rows=1, fps=fps)

            # 添加音频
            video_clip = VideoFileClip(str(output_video_path))
            audio_clip = AudioFileClip(audio_path)
            final_output_path = output_dir / f"{task_id}_segment_{idx}_with_audio.mp4"
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac")
            
            output_files.append(str(final_output_path))
            video_clips.append(video_clip)
        
        # 如果需要合并视频
        combined_output_file = None
        if combine_videos and len(video_clips) > 1:
            final_clip = concatenate_videoclips(video_clips)
            combined_output_path = output_dir / f"{task_id}_combined.mp4"
            final_clip.write_videofile(str(combined_output_path), codec="libx264", audio_codec="aac")
            combined_output_file = str(combined_output_path)
        
        # 清理临时文件
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # 关闭所有视频片段
        for clip in video_clips:
            clip.close()
        
        # 更新任务状态为完成
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["output_files"] = output_files
        if combined_output_file:
            tasks[task_id]["combined_output_file"] = combined_output_file
        
        # 释放显存
        torch.cuda.empty_cache()
        
    except Exception as e:
        # 发生错误，更新任务状态
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        print(f"任务 {task_id} 处理失败: {e}")
        torch.cuda.empty_cache()

@app.get("/")
async def root():
    """API 根路径"""
    return {"message": "欢迎使用 EchoMimic API", "version": "1.0"}

@app.post("/generate_video")
async def generate_video(
    background_tasks: BackgroundTasks,
    reference_image: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    width: int = Form(default_values["width"]),
    height: int = Form(default_values["height"]),
    length: int = Form(default_values["length"]),
    seed: int = Form(default_values["seed"]),
    facemask_dilation_ratio: float = Form(default_values["facemask_dilation_ratio"]),
    facecrop_dilation_ratio: float = Form(default_values["facecrop_dilation_ratio"]),
    context_frames: int = Form(default_values["context_frames"]),
    context_overlap: int = Form(default_values["context_overlap"]),
    cfg: float = Form(default_values["cfg"]),
    steps: int = Form(default_values["steps"]),
    sample_rate: int = Form(default_values["sample_rate"]),
    fps: int = Form(default_values["fps"]),
    device: str = Form(default_values["device"])
):
    """生成视频接口"""
    # 创建唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 保存上传的文件
    img_path = temp_dir / f"{task_id}_reference.png"
    audio_path = temp_dir / f"{task_id}_audio.wav"
    
    with open(img_path, "wb") as img_file:
        img_file.write(await reference_image.read())
    
    with open(audio_path, "wb") as audio_file_obj:
        audio_file_obj.write(await audio_file.read())
    
    # 创建任务记录
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "output_file": None,
        "error": None
    }
    
    # 启动异步处理任务
    background_tasks.add_task(
        process_video_task,
        task_id,
        str(img_path),
        str(audio_path),
        width,
        height,
        length,
        seed,
        facemask_dilation_ratio,
        facecrop_dilation_ratio,
        context_frames,
        context_overlap,
        cfg,
        steps,
        sample_rate,
        fps,
        device
    )
    
    return {"task_id": task_id, "status": "pending"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]

@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """下载生成的视频"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务状态: {task['status']}, 不可下载")
    
    output_file = task["output_file"]
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    return FileResponse(
        path=output_file,
        filename=f"echomimic_{task_id}.mp4",
        media_type="video/mp4"
    )

@app.post("/generate_multi_videos")
async def generate_multi_videos(
    background_tasks: BackgroundTasks,
    reference_image: UploadFile = File(...),
    audio_files: List[UploadFile] = File(...),
    width: int = Form(default_values["width"]),
    height: int = Form(default_values["height"]),
    length: int = Form(default_values["length"]),
    seed: int = Form(default_values["seed"]),
    facemask_dilation_ratio: float = Form(default_values["facemask_dilation_ratio"]),
    facecrop_dilation_ratio: float = Form(default_values["facecrop_dilation_ratio"]),
    context_frames: int = Form(default_values["context_frames"]),
    context_overlap: int = Form(default_values["context_overlap"]),
    cfg: float = Form(default_values["cfg"]),
    steps: int = Form(default_values["steps"]),
    sample_rate: int = Form(default_values["sample_rate"]),
    fps: int = Form(default_values["fps"]),
    device: str = Form(default_values["device"]),
    combine_videos: bool = Form(False)
):
    """生成多段视频接口"""
    # 创建唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 保存上传的参考图像
    img_path = temp_dir / f"{task_id}_reference.png"
    with open(img_path, "wb") as img_file:
        img_file.write(await reference_image.read())
    
    # 保存所有的音频文件
    audio_paths = []
    for idx, audio_file in enumerate(audio_files):
        audio_path = temp_dir / f"{task_id}_audio_{idx}.wav"
        with open(audio_path, "wb") as audio_file_obj:
            audio_file_obj.write(await audio_file.read())
        audio_paths.append(str(audio_path))
    
    # 创建任务记录
    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "output_files": None,
        "combined_output_file": None,
        "error": None
    }
    
    # 启动异步处理任务
    background_tasks.add_task(
        process_multi_audio_video_task,
        task_id,
        str(img_path),
        audio_paths,
        width,
        height,
        length,
        seed,
        facemask_dilation_ratio,
        facecrop_dilation_ratio,
        context_frames,
        context_overlap,
        cfg,
        steps,
        sample_rate,
        fps,
        device,
        combine_videos
    )
    
    return {"task_id": task_id, "status": "pending"}

@app.get("/download_segment/{task_id}/{segment_id}")
async def download_video_segment(task_id: str, segment_id: int):
    """下载生成的视频片段"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务状态: {task['status']}, 不可下载")
    
    if "output_files" not in task or not task["output_files"]:
        raise HTTPException(status_code=404, detail="没有可用的视频片段")
    
    if segment_id < 0 or segment_id >= len(task["output_files"]):
        raise HTTPException(status_code=404, detail=f"片段 {segment_id} 不存在")
    
    output_file = task["output_files"][segment_id]
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    return FileResponse(
        path=output_file,
        filename=f"echomimic_{task_id}_segment_{segment_id}.mp4",
        media_type="video/mp4"
    )

@app.get("/download_combined/{task_id}")
async def download_combined_video(task_id: str):
    """下载合并后的视频"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务状态: {task['status']}, 不可下载")
    
    if "combined_output_file" not in task or not task["combined_output_file"]:
        raise HTTPException(status_code=404, detail="没有合并的视频文件")
    
    output_file = task["combined_output_file"]
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    return FileResponse(
        path=output_file,
        filename=f"echomimic_{task_id}_combined.mp4",
        media_type="video/mp4"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EchoMimic FastAPI 服务')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机名')
    parser.add_argument('--port', type=int, default=7688, help='服务器端口')
    args = parser.parse_args()
    
    print(f"启动 EchoMimic FastAPI 服务，地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port) 