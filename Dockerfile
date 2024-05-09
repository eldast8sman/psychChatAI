# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    OPENAI_API_KEY=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

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["flask", "run"]
