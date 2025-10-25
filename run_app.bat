@echo off
echo Starting Microplastic Detection Software...
echo ==========================================
cd /d "D:\my project"
C:\Users\Thejukusuma\AppData\Local\Programs\Python\Python312\python.exe -m streamlit run streamlit_app_enhanced.py --server.port 8501 --server.address localhost
pause
