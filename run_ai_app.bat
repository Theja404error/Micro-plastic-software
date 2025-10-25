@echo off
echo Starting AI-Powered Microplastic Detection Software...
echo ===================================================
cd /d "D:\my project"
C:\Users\Thejukusuma\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run ai_microplastic_detector.py --server.port 8502 --server.address localhost
pause
