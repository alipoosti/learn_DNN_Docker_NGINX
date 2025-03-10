# Use the official Python image as the base image
# FROM python:3.9
# FROM nvcr.io/nvidia/pytorch:23.08-py3
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /my_inference_app

# Copy requirements into the container
COPY  requirements.txt .

# Install requirements
RUN pip install -U pip
RUN pip install --no-cache-dir --default-timeout=900 -r requirements.txt

# Expose the port the app runs on
# EXPOSE 8000

# Copy all required files into the container
COPY . .

# Define the command to run the Python script
# CMD ["python3", "wsgi.py"]