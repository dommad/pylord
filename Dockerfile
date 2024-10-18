# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

RUN mkdir /usr/src/app/ext

# Copy the requirements file into the container at /usr/src/app
COPY requirements.txt ./

# Install GCC and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc


# Install any dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /usr/src/app
COPY . .

# Ensure the modules in the pylord folder can be executed
ENTRYPOINT ["python", "-m"]
