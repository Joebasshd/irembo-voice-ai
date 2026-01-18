# 1. Start with a lightweight Python image
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your project files into the container
# We only copy what is needed for the API to run
COPY ./src /app/src
COPY ./models /app/models
COPY ./data /app/data

# 6. Expose the port FastAPI will run on
EXPOSE 8000

# 7. Start the FastAPI server using uvicorn
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]