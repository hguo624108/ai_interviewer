@echo off
chcp 65001 >nul
echo Starting Virtual Interviewer Microservice System...

echo 1. Starting CosyVoice service...
start cmd /k "cd CosyVoice/runtime/python/fastapi && conda activate Cosyvoice1 && python server_with_azure.py"

echo 2. Starting EchoMimic service...
start cmd /k "cd Echomimic && run_api.bat"

echo 3. Waiting for services to start...
timeout /t 60

echo 4. Starting Virtual Interviewer API...
python main.py

pause