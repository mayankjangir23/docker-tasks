# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run training during build
RUN python train_model.py

# Default command
ENTRYPOINT ["python", "predict.py"]
