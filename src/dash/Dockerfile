FROM python:3.7.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000

COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:server"]
