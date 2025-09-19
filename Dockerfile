# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy all project files
COPY . .

# # Copy .env file
# COPY .env /app/.env

# Expose port (default 8000)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
