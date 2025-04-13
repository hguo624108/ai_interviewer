#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Virtual Interviewer API - Microservice application integrating CosyVoice and EchoMimic
'''

import os
import uuid
import json
import shutil
import asyncio
import time
import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import requests
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
API_CONFIG = {
    "cosyvoice": "http://localhost:8000",  # CosyVoice service address
    "echomimic": "http://localhost:7688",  # EchoMimic service address
}

# Video generation parameters
VIDEO_PARAMS = {
    'width': 512,
    'height': 512,
    'steps': 6,
    'context_frames': 12,
    'context_overlap': 3,
    'facemask_dilation_ratio': 0.1,
    'facecrop_dilation_ratio': 0.7,
    'cfg': 1,
    'fps': 24,
    'sample_rate': 16000,
    'combine_videos': 'false'
}

# Create directories
BASE_DIR = Path(__file__).parent.absolute()
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = BASE_DIR / "results"
TEMP_DIR = BASE_DIR / "temp"
AUDIO_DIR = TEMP_DIR / "audio"

# Ensure directories exist
for directory in [STATIC_DIR, RESULTS_DIR, TEMP_DIR, AUDIO_DIR,
                 STATIC_DIR / "images", STATIC_DIR / "voices"]:
    directory.mkdir(exist_ok=True, parents=True)

# Default resources
DEFAULT_IMAGES = {
    "male_interviewer": str(STATIC_DIR / "images" / "male_interviewer.png"),
    "female_interviewer": str(STATIC_DIR / "images" / "female_interviewer.png")
}

DEFAULT_VOICES = {
    "male_voice": str(STATIC_DIR / "voices" / "male_voice.wav"),
    "female_voice": str(STATIC_DIR / "voices" / "female_voice.wav")
}

# Create FastAPI application
app = FastAPI(
    title="Virtual Interviewer API",
    description="Virtual interviewer service integrating CosyVoice and EchoMimic",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directories
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Define data models
class Question(BaseModel):
    text: str = Field(..., description="Interview question text")

class CustomInterviewRequest(BaseModel):
    questions: List[Question] = Field(..., max_items=10, description="List of interview questions, maximum 10")

class DefaultInterviewRequest(BaseModel):
    questions: List[Question] = Field(..., max_items=10, description="List of interview questions, maximum 10")
    image_type: str = Field(..., description="Default avatar type", example="male_interviewer")
    voice_type: str = Field(..., description="Default voice type", example="male_voice")

class InterviewTask(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    questions: List[str]
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

# Store task status
interview_tasks = {}

# Helper functions
@contextmanager
def safe_open_file(file_path, mode='rb'):
    """Context manager for safely opening files"""
    file = None
    try:
        file = open(file_path, mode)
        yield file
    finally:
        if file:
            file.close()

def validate_questions(questions: List[str]) -> List[str]:
    """Validate and filter the question list"""
    if len(questions) > 10:
        raise HTTPException(status_code=400, detail="A maximum of 10 questions can be submitted")
    
    filtered = [q for q in questions if q.strip()]
    
    if not filtered:
        raise HTTPException(status_code=400, detail="At least one question must be provided")
    
    return filtered

async def save_uploaded_file(upload_file: UploadFile, destination: Path) -> None:
    """Save an uploaded file"""
    with open(destination, "wb") as f:
        f.write(await upload_file.read())

def create_task_record(task_id: str, questions: List[str], image_path: str, voice_path: str) -> Dict:
    """Create a task record"""
    return {
        "task_id": task_id,
        "status": "pending",
        "questions": questions,
        "image_path": image_path,
        "voice_path": voice_path,
        "results": [],
        "error": None
    }

# Routes
@app.get("/")
async def root():
    return {"message": "Virtual Interviewer API", "version": "1.0"}

@app.post("/interview/custom")
async def create_custom_interview(
    background_tasks: BackgroundTasks,
    questions: List[str] = Form(...),
    image: UploadFile = File(...),
    voice: UploadFile = File(...)
):
    """
    Create a custom interviewer task
    - **questions**: List of interview questions (maximum 10)
    - **image**: Interviewer avatar image
    - **voice**: Interviewer voice sample audio
    """
    logger.info(f"Received custom interview request: {len(questions)} questions")
    
    # Validate and filter questions
    filtered_questions = validate_questions(questions)
    
    # Validate file format
    image_extension = image.filename.split('.')[-1].lower()
    voice_extension = voice.filename.split('.')[-1].lower()
    
    if image_extension not in ['png', 'jpg', 'jpeg']:
        raise HTTPException(status_code=400, detail="Image format must be PNG or JPG")
    
    if voice_extension not in ['wav']:
        raise HTTPException(status_code=400, detail="Audio format must be WAV")
    
    # Create task directory and ID
    task_id = str(uuid.uuid4())
    task_dir = TEMP_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    image_path = task_dir / f"image.{image_extension}"
    voice_path = task_dir / f"voice.{voice_extension}"
    
    await save_uploaded_file(image, image_path)
    await save_uploaded_file(voice, voice_path)
    
    # Create task record
    interview_tasks[task_id] = create_task_record(
        task_id, filtered_questions, str(image_path), str(voice_path)
    )
    
    # Start background task processing
    background_tasks.add_task(process_interview_task, task_id)
    
    return {"task_id": task_id, "status": "pending"}

@app.post("/interview/default")
async def create_default_interview(
    background_tasks: BackgroundTasks,
    questions: List[str] = Form(...),
    image_type: str = Form(...),
    voice_type: str = Form(...)
):
    """
    Create a default interviewer task
    - **questions**: List of interview questions (maximum 10)
    - **image_type**: Default avatar type (male_interviewer/female_interviewer)
    - **voice_type**: Default voice type (male_voice/female_voice)
    """
    logger.info(f"Received default interview request: {len(questions)} questions, avatar={image_type}, voice={voice_type}")
    
    # Validate and filter questions
    filtered_questions = validate_questions(questions)
    
    # Validate default resources
    if image_type not in DEFAULT_IMAGES:
        raise HTTPException(status_code=400, detail=f"Invalid avatar type: {image_type}")
    
    if voice_type not in DEFAULT_VOICES:
        raise HTTPException(status_code=400, detail=f"Invalid voice type: {voice_type}")
    
    # Create task ID
    task_id = str(uuid.uuid4())
    
    # Create task record
    interview_tasks[task_id] = create_task_record(
        task_id, filtered_questions, DEFAULT_IMAGES[image_type], DEFAULT_VOICES[voice_type]
    )
    
    # Start background task processing
    background_tasks.add_task(process_interview_task, task_id)
    
    return {"task_id": task_id, "status": "pending"}

@app.get("/interview/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status and results of an interview task"""
    if task_id not in interview_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = interview_tasks[task_id]
    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "questions": task["questions"],
        "results": task["results"],
        "error": task["error"]
    }

@app.get("/interview/video/{video_id}")
async def get_video(video_id: str):
    """Get the generated video file"""
    video_path = RESULTS_DIR / f"{video_id}.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=str(video_path),
        filename=f"interview_{video_id}.mp4",
        media_type="video/mp4"
    )

@app.get("/test")
async def test_ui():
    """Test interface"""
    return FileResponse(str(STATIC_DIR / "test_ui.html"), media_type="text/html")

async def process_interview_task(task_id: str):
    """Background function for processing interview tasks"""
    task = interview_tasks[task_id]
    task["status"] = "processing"
    logger.info(f"Starting to process task {task_id}: {len(task['questions'])} questions")
    
    try:
        results = []
        
        # Phase 1: Generate audio for all questions
        logger.info(f"Task {task_id}: Starting to generate all audio...")
        audio_results = []
        
        for i, question in enumerate(task["questions"]):
            logger.info(f"Task {task_id}: Generating audio for question {i+1}/{len(task['questions'])}")
            audio_result = await generate_audio(task["voice_path"], question, task_id, i)
            
            status = "success" if "audio_path" in audio_result else "failed"
            audio_results.append({
                "question": question,
                "status": status,
                "error": audio_result.get("error"),
                "audio_path": audio_result.get("audio_path")
            })
        
        # Phase 2: Generate videos for all audio
        logger.info(f"Task {task_id}: Starting to generate all videos...")
        
        # Collect valid audio paths
        valid_audio_paths = []
        valid_audio_indices = []
        
        for i, audio_result in enumerate(audio_results):
            if audio_result["status"] == "success" and audio_result["audio_path"]:
                valid_audio_paths.append(audio_result["audio_path"])
                valid_audio_indices.append(i)
            else:
                # Record failed results
                results.append({
                    "question": task["questions"][i],
                    "status": "failed",
                    "error": audio_result.get("error", "Audio generation failed"),
                    "video_id": None,
                    "video_url": None
                })
        
        # If there are valid audio files, batch generate videos
        if valid_audio_paths:
            logger.info(f"Task {task_id}: Using multi-audio API to batch generate {len(valid_audio_paths)} videos...")
            video_result = await generate_multi_videos(task["image_path"], valid_audio_paths)
            
            if "error" in video_result:
                # All video generation failed
                for i in valid_audio_indices:
                    results.append({
                        "question": task["questions"][i],
                        "status": "failed",
                        "error": video_result["error"],
                        "video_id": None,
                        "video_url": None
                    })
            elif "video_paths" in video_result and video_result["video_paths"]:
                # Process successfully generated videos
                for idx, path_idx in enumerate(valid_audio_indices):
                    if idx < len(video_result["video_paths"]):
                        video_id = f"{task_id}_{path_idx}"
                        video_path = video_result["video_paths"][idx]
                        final_video_path = RESULTS_DIR / f"{video_id}.mp4"
                        
                        if os.path.exists(video_path):
                            # Copy generated video to results directory
                            shutil.copy(video_path, final_video_path)
                            
                            results.append({
                                "question": task["questions"][path_idx],
                                "status": "completed",
                                "video_id": video_id,
                                "video_url": f"/interview/video/{video_id}"
                            })
                        else:
                            results.append({
                                "question": task["questions"][path_idx],
                                "status": "failed",
                                "error": f"Video file does not exist: {video_path}",
                                "video_id": None,
                                "video_url": None
                            })
        else:
            logger.info(f"Task {task_id}: No valid audio files, skipping video generation step")
        
        # Update task status
        task["results"] = results
        task["status"] = "completed"
        logger.info(f"Task {task_id} processing completed")
        
        # Clean up temporary files
        clean_temp_files(task_id)
            
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}", exc_info=True)
        task["status"] = "failed"
        task["error"] = str(e)

def clean_temp_files(task_id: str) -> None:
    """Clean up temporary files for a task"""
    temp_task_dir = TEMP_DIR / task_id
    if temp_task_dir.exists():
        try:
            for file_path in temp_task_dir.glob("*"):
                if not str(file_path).endswith((".wav", ".mp4")):
                    os.remove(file_path)
            logger.info(f"Cleaned up temporary files for task {task_id}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")

async def generate_audio(voice_path: str, text: str, task_id: str, question_index: int) -> Dict:
    """Call CosyVoice API to generate audio"""
    try:
        # Prepare output directory and filename
        output_dir = str(AUDIO_DIR / task_id)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"question_{question_index}"
        
        # Prepare request data
        with safe_open_file(voice_path, 'rb') as f:
            files = {
                'prompt_audio': ('voice.wav', f, 'audio/wav')
            }
            
            data = {
                'texts': text,
                'output_dir': output_dir,
                'output_filenames': output_filename,
                'return_type': 'path',
                'input_format': 'text'
            }
            
            # Send request to CosyVoice service
            logger.info(f"Sending request to CosyVoice: text length={len(text)}")
            response = requests.post(f"{API_CONFIG['cosyvoice']}/cross_lingual_batch", files=files, data=data)
        
        if response.status_code != 200:
            return {"error": f"CosyVoice service error: {response.text}"}
        
        # Parse response
        result = response.json()
        
        if result["status"] != "success" or not result["results"]:
            return {"error": "Audio generation failed"}
        
        # Get generated audio file path
        audio_result = result["results"][0]
        if audio_result["status"] != "success":
            return {"error": f"Audio generation failed: {audio_result.get('error', 'Unknown error')}"}
        
        return {"audio_path": audio_result["output_path"]}
    
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}", exc_info=True)
        return {"error": f"Error generating audio: {str(e)}"}

async def generate_multi_videos(image_path: str, audio_paths: List[str]) -> Dict:
    """Call EchoMimic multi-audio API to generate multiple videos"""
    try:
        logger.info(f"Starting multi-video generation: audio count={len(audio_paths)}")
        
        # Ensure files exist
        if not os.path.exists(image_path):
            return {"error": f"Image file does not exist: {image_path}"}
        
        for audio_path in audio_paths:
            if not os.path.exists(audio_path):
                return {"error": f"Audio file does not exist: {audio_path}"}
        
        # Prepare multi-file upload
        files = []
        opened_files = []
        
        try:
            # Add image
            image_file = open(image_path, 'rb')
            opened_files.append(image_file)
            files.append(('reference_image', ('image.png', image_file, 'image/png')))
            
            # Add multiple audio files
            for audio_path in audio_paths:
                audio_file = open(audio_path, 'rb')
                opened_files.append(audio_file)
                files.append(('audio_files', ('audio.wav', audio_file, 'audio/wav')))
            
            # Send request to EchoMimic service
            logger.info(f"Sending request to EchoMimic: audio count={len(audio_paths)}")
            response = requests.post(
                f"{API_CONFIG['echomimic']}/generate_multi_videos", 
                files=files, 
                data=VIDEO_PARAMS
            )
        finally:
            # Close all opened files
            for f in opened_files:
                f.close()
        
        if response.status_code != 200:
            return {"error": f"EchoMimic service error: {response.text}"}
        
        # Parse response to get task ID
        result = response.json()
        if "task_id" not in result:
            return {"error": f"Invalid EchoMimic response format: {result}"}
            
        task_id = result["task_id"]
        logger.info(f"EchoMimic multi-video task created: {task_id}")
        
        # Wait for task completion
        return await wait_for_echomimic_task(task_id)
    
    except Exception as e:
        logger.error(f"Error generating multiple videos: {str(e)}", exc_info=True)
        return {"error": f"Error generating multiple videos: {str(e)}"}

async def wait_for_echomimic_task(task_id: str, max_retries: int = 240, retry_interval: int = 5) -> Dict:
    """Wait for EchoMimic task to complete and download the results"""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Check task status
            status_response = requests.get(f"{API_CONFIG['echomimic']}/task/{task_id}")
            if status_response.status_code != 200:
                logger.warning(f"Failed to get task status, status code: {status_response.status_code}")
                retry_count += 1
                await asyncio.sleep(retry_interval)
                continue
                
            status = status_response.json()
            logger.info(f"Task {task_id} status: {status['status']}")
            
            # Handle completed status
            if status["status"] == "completed":
                return await download_echomimic_results(task_id, status)
            
            # Handle failed status
            elif status["status"] == "failed":
                error_msg = status.get("error", "Unknown error")
                logger.error(f"Video generation failed: {error_msg}")
                return {"error": f"Video generation failed: {error_msg}"}
            
            # Continue waiting
            await asyncio.sleep(retry_interval)
            retry_count += 1
            
        except Exception as e:
            logger.error(f"Error checking task status: {e}")
            retry_count += 1
            await asyncio.sleep(retry_interval)
    
    return {"error": "Video generation timeout, please check EchoMimic service status"}

async def download_echomimic_results(task_id: str, status: Dict) -> Dict:
    """Download the resulting videos from an EchoMimic task"""
    result = {"task_id": task_id, "video_paths": []}
    
    # Download each video segment
    if "output_files" in status and status["output_files"]:
        for i in range(len(status["output_files"])):
            segment_path = TEMP_DIR / f"{task_id}_segment_{i}.mp4"
            
            segment_response = requests.get(f"{API_CONFIG['echomimic']}/download_segment/{task_id}/{i}", stream=True)
            if segment_response.status_code == 200:
                with open(segment_path, "wb") as f:
                    for chunk in segment_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                result["video_paths"].append(str(segment_path))
                logger.info(f"Video segment {i} downloaded to: {segment_path}")
            else:
                logger.error(f"Video segment {i} download failed, status code: {segment_response.status_code}")
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)