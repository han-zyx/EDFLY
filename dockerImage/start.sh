#!/bin/bash

# Ensure environment variables are set
if [[ -z "$KAFKA_HOME" || -z "$FLINK_HOME" ]]; then
    echo "Error: KAFKA_HOME or FLINK_HOME is not set. Exiting."
    exit 1
fi

# Start Zookeeper
echo "Starting Zookeeper..."
nohup $KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties > zookeeper.log 2>&1 &

# Wait for Zookeeper to start
sleep 5

# Start Kafka
echo "Starting Kafka..."
nohup $KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties > kafka.log 2>&1 &

# Wait for Kafka to start
sleep 10

# Create Kafka topic (Check if it already exists)
echo "Creating Kafka topic..."
$KAFKA_HOME/bin/kafka-topics.sh --create --topic input-topic --bootstrap-server 0.0.0.0:9092 --partitions 1 --replication-factor 1 || true

# Start Flink cluster
echo "Starting Flink..."
nohup $FLINK_HOME/bin/start-cluster.sh > flink.log 2>&1 &

# Wait for Flink to start
sleep 10

# Submit the job
echo "Submitting Flink job..."
$FLINK_HOME/bin/flink run -c KafkaFlinkJsonIntegration ../flink-yolov8-demo/target/flink-kafka-demo-1.0.jar

echo "All services started and job submitted successfully."

# Keep the container alive
tail -f /dev/null
