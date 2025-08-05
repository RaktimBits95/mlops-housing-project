# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy everything from your local folder into /app in the container
COPY . /app

# Install dependencies required for inference
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas joblib pydantic

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
