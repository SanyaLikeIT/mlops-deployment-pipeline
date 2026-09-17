FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY code/deployment/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY code/deployment/app/app.py ./app.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
