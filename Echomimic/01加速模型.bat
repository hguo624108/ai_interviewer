@echo off
chcp 65001 >nul
echo  启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892

set PYTHON=%CD%\.glut\python.exe
set CU_PATH=%CD%\.glut\Lib\site-packages\torch\lib
set SC_PATH=%CD%\.glut\Scripts
set FFMPEG_PATH=%CD%\.glut\ffmpeg\bin
set PATH=%FF_PATH%;%CU_PATH%;%SC_PATH%;%FFMPEG_PATH%;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\.huggingface
set XFORMERS_FORCE_DISABLE_TRITON=1


%PYTHON% webgui_glut.py

pause

