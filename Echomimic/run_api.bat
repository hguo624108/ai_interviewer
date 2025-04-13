@echo off
chcp 65001 >nul
echo 正在启动 EchoMimic FastAPI 服务，请耐心等待...

set PYTHON=%CD%\.glut\python.exe
set CU_PATH=%CD%\.glut\Lib\site-packages\torch\lib
set SC_PATH=%CD%\.glut\Scripts
set FFMPEG_PATH=%CD%\.glut\ffmpeg\bin
set PATH=%FF_PATH%;%CU_PATH%;%SC_PATH%;%FFMPEG_PATH%;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\.huggingface
set XFORMERS_FORCE_DISABLE_TRITON=1

%PYTHON% fastapi_server.py

pause 