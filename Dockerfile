# Use the official Python image as a base
FROM ubuntu:24.04

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# install apt packages
RUN apt-get update
RUN apt-get install -y build-essential python3 python3-pip python3-venv

RUN python3 -m venv .venv
ENV PATH=".venv/bin:$PATH"

# Install the required Python packages
RUN pip install -U --upgrade pip
RUN pip install -U --no-cache-dir -r requirements.txt

# Copy the FastAPI app code into the container
COPY . /app/

# Expose the application port (FastAPI defaults to 8000)
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]