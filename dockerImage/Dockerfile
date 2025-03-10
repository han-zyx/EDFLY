# Use an official Python base image
FROM python:3.9

# Set environment variables to avoid interactive installation issues
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk-headless netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Flink and Kafka from your local machine into the container
COPY flink /app/flink
COPY kafka /app/kafka

# Set environment variables
ENV KAFKA_HOME=/app/kafka
ENV FLINK_HOME=/app/flink
ENV PATH="$KAFKA_HOME/bin:$FLINK_HOME/bin:$PATH"

# Copy application files
COPY . .

# Install Python dependencies (if any)
RUN pip install --no-cache-dir -r requirements.txt || true

# Expose necessary ports
EXPOSE 9092 2181 8081 6123

# Make the startup script executable
RUN chmod +x start.sh

# Run Kafka, Zookeeper, Flink, and your Python app
CMD ["./start.sh"]
