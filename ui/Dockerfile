FROM python:3.11-slim
WORKDIR /ui

COPY ui/ui_requirements.txt ./ui_requirements.txt
RUN pip install --no-cache-dir -r ui_requirements.txt

COPY ui/streamlit_app.py ./streamlit_app.py

# 도커 네트워크에서 백엔드 서비스명(api)으로 접근
ENV API_BASE=http://api:8000

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
