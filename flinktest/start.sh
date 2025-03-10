#!/bin/bash

echo "hello world"

# # Ensure FLINK_HOME is set
# if [[ -z "$FLINK_HOME" ]]; then
#     echo "Error: FLINK_HOME is not set. Please set the FLINK_HOME environment variable. Exiting."
#     exit 1
# fi

# # Start Flink cluster
# echo "Starting Flink cluster..."
# nohup "$FLINK_HOME/bin/start-cluster.sh" > flink-startup.log 2>&1 &

# # Wait for Flink to start (adjust sleep duration if needed)
# echo "Waiting for Flink to start..."
# sleep 10

# # Check if Flink cluster started successfully
# if ! pgrep -f "flink" > /dev/null; then
#     echo "Error: Flink cluster failed to start. Check flink-startup.log for details. Exiting."
#     exit 1
# fi

# # Submit the Flink job (replace with your actual job or modify for Python jobs)
# JAR_PATH="/app/flink-jobs/your-flink-job.jar"
# MAIN_CLASS="KafkaFlinkJsonIntegration"

# if [[ ! -f "$JAR_PATH" ]]; then
#     echo "Error: Flink job JAR file not found at $JAR_PATH. Exiting."
#     exit 1
# fi

# echo "Submitting Flink job..."
# "$FLINK_HOME/bin/flink" run -c "$MAIN_CLASS" "$JAR_PATH"

# if [[ $? -eq 0 ]]; then
#     echo "Flink job submitted successfully."
# else
#     echo "Error: Flink job submission failed. Check the logs for details. Exiting."
#     exit 1
# fi

# # Keep the container alive (useful for Docker containers)
# echo "Keeping the container alive..."
# tail -f /dev/null
