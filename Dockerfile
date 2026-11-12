FROM python:3.12-slim
WORKDIR /app
ENV AXON_BOOTSTRAPPED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=5000
COPY requirements.txt requirements-teacher.txt ./
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-teacher.txt
COPY . .
EXPOSE 5000
CMD ["python","HTML.py"]
