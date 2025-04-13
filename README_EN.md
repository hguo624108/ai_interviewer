# Text-to-Video Generation Service

This project implements a text-to-video generation service that integrates CosyVoice voice cloning and EchoMimic facial animation technologies to convert text into videos featuring specified characters speaking. The service supports multiple text segments, with each segment generating an independent video.

## Project Structure

```
proj/
│
├── main.py          # FastAPI main program
├── requirements.txt     # Main program dependencies
├── run_services.bat      # Batch startup file
│
├── CosyVoice/           # CosyVoice model directory
│   ├── Cosyvoice1/      # CosyVoice local Python environment
│   ├── cosyvoice_with_azure.py  # CosyVoice model implementation
│   └── pretrained_models/       # Pre-trained model directory
│
├── Echomimic/           # EchoMimic model directory
│   ├── .glut/           # EchoMimic local Python environment
│   └── webgui_glut.py   # EchoMimic model implementation
│
├── static/              # Static files directory
│   └── outputs/         # Output files storage directory
│
└── temp/                # Temporary files directory
```

## Environmental Requirements

### Main Program Environment

The main program uses Python 3.8+ and depends on the following libraries:
- FastAPI
- Uvicorn
- Python-multipart
- Pydantic
- Other dependencies (see requirements.txt for details)

You can install the main program dependencies with the following command:
```bash
pip install -r requirements.txt
```

### CosyVoice Environment

The CosyVoice model supports two environment configuration methods:

1. Using local pre-configured environment (recommended):
   - Path: `./CosyVoice/Cosyvoice1/`
   - Already includes all necessary dependencies and configurations
   - Ready to use without additional setup

2. Creating your own virtual environment:
   - Create a new Python virtual environment
   - Install necessary dependencies:
     - torch
     - torchaudio
     - azure-cognitiveservices-speech
     - Other dependencies (see CosyVoice/requirements.txt for details)

### EchoMimic Environment

The EchoMimic model uses a local Python environment:
- Path: `./Echomimic/.glut/python.exe`
- Main dependencies: torch, diffusers, omegaconf, PIL, cv2, facenet-pytorch, etc.

## Starting the Service

Use the following command to start the service:

```bash
python main_new.py
```

Or run the batch file directly:

```bash
run_service.bat
```

The service will run at http://localhost:9000.

## API Documentation

### 1. Text-to-Video Conversion

**Request**:
- URL: `/text-to-video/`
- Method: `POST`
- Form Data:
  - `texts`: List of texts to convert (can contain multiple text segments)
  - `prompt_audio`: Prompt audio file (for voice cloning)
  - `reference_image`: Reference image file (for facial animation)

**Response**:
```json
{
  "task_id": "uuid-string",
  "status": "processing",
  "total_segments": 3
}
```

### 2. Get Task Status

**Request**:
- URL: `/tasks/{task_id}`
- Method: `GET`

**Response**:
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "duration": 350.5,
  "video_urls": [
    "/static/outputs/uuid-string/video_uuid-string_1.mp4",
    "/static/outputs/uuid-string/video_uuid-string_2.mp4",
    "/static/outputs/uuid-string/video_uuid-string_3.mp4"
  ],
  "total_segments": 3,
  "progress": 100.0
}
```

Possible status values:
- `processing`: In progress
- `generating_audio_1`, `generating_audio_2`...: Generating audio for specific text segment
- `generating_video_1`, `generating_video_2`...: Generating video for specific text segment
- `completed`: Completed
- `failed`: Failed

### 3. Get Individual Video File

**Request**:
- URL: `/video/{task_id}/{segment}`
- Method: `GET`
- Parameters:
  - `segment`: Segment number, starting from 1

**Response**: Video file download

### 4. Get All Video Information

**Request**:
- URL: `/videos/{task_id}`
- Method: `GET`

**Response**:
```json
{
  "task_id": "uuid-string",
  "total_segments": 3,
  "videos": [
    {
      "segment": 1,
      "video_url": "/static/outputs/uuid-string/video_uuid-string_1.mp4",
      "download_url": "/video/uuid-string/1"
    },
    {
      "segment": 2,
      "video_url": "/static/outputs/uuid-string/video_uuid-string_2.mp4",
      "download_url": "/video/uuid-string/2"
    },
    {
      "segment": 3,
      "video_url": "/static/outputs/uuid-string/video_uuid-string_3.mp4",
      "download_url": "/video/uuid-string/3"
    }
  ]
}
```

## Processing Flow

1. Receive multiple text segments and reference audio/image
2. Process each text segment sequentially:
   a. Use the CosyVoice model (cross_lingual mode) to generate speech with a voice similar to the reference audio
   b. Use the EchoMimic model to generate talking video based on the generated speech and reference image
3. Return all generated video files

## Resource Management

- Models are loaded as needed during processing
- Resources are released after each text segment is processed
- Linear processing approach: complete the entire process for one text segment before moving to the next
- GPU memory is cleared after each phase

## Implementation Details

- The main program uses relative paths for better portability
- Auxiliary scripts are automatically generated at startup, no manual maintenance needed
- The two models run in their own independent Python environments, avoiding dependency conflicts
- Data is transferred through inter-process communication, ensuring resource isolation
- Detailed progress information is provided, including the current segment being processed and overall progress percentage

## Notes

1. Ensure the local environment paths for CosyVoice and EchoMimic are correct
2. Ensure pre-trained model files are correctly installed
3. For the prompt audio, it's recommended to use 5-10 seconds of clear human voice
4. For the reference image, it's recommended to use a clear front-facing facial image
5. Processing large videos or multiple text segments may take a long time, please be patient
6. The more text segments, the more processing time and resources required 
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
   - For CosyVoice: Create a Python environment in `CosyVoice/Cosyvoice1/`
   - For EchoMimic: Create a Python environment in `Echomimic/.glut/`

3. **Create directories**:
   - Create directories for `results/` and `temp/` for storing generated files

Please refer to the installation instructions in the documentation for more detailed setup guidance.

