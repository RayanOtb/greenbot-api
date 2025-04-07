FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with memory optimization
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tensorflow-cpu==2.15.0 && \
    pip cache purge

# Copy the rest of the application
COPY . .

# Run setup_model.py to create the model
RUN python setup_model.py

# Set TensorFlow environment variables for CPU optimization
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV TF_XLA_FLAGS=--tf_xla_cpu_global_jit

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application with memory limits
CMD ["python", "main.py"] 