#!/bin/bash

# Set Kafka options
export KAFKA_OPTS="-javaagent:/home/hansaka/fyp/EDFLY/kafka/libs/jmx_prometheus_javaagent-0.16.1.jar=9093:/home/hansaka/fyp/EDFLY/kafka/kafka-jmx-exporter.yaml"

# Define base directory for Kafka and Flink (adjust if needed)
KAFKA_HOME="/home/hansaka/fyp/EDFLY/kafka"
FLINK_HOME="/home/hansaka/fyp/EDFLY/flink"

# Start Zookeeper in the background
echo "Starting Zookeeper..."
${KAFKA_HOME}/bin/zookeeper-server-start.sh ${KAFKA_HOME}/config/zookeeper.properties > zookeeper.log 2>&1 &
ZOOKEEPER_PID=$!
sleep 5  # Give Zookeeper some time to start

# Start Kafka server in the background
echo "Starting Kafka Server..."
${KAFKA_HOME}/bin/kafka-server-start.sh ${KAFKA_HOME}/config/server.properties > kafka.log 2>&1 &
KAFKA_PID=$!
sleep 5  # Give Kafka some time to start

# Create Kafka topic
echo "Creating Kafka topic 'input-topic'..."
${KAFKA_HOME}/bin/kafka-topics.sh --create --topic input-topic --bootstrap-server localhost:9092

# Start Flink cluster
echo "Starting Flink Cluster..."
${FLINK_HOME}/bin/start-cluster.sh > flink.log 2>&1 &
FLINK_PID=$!
sleep 5  # Give Flink some time to start

# Submit the Flink job
echo "Submitting Flink job..."
${FLINK_HOME}/bin/flink run -c KafkaFlinkJsonIntegration ../flink_jar/target/flink-kafka-demo-1.0.jar

echo "All services started. Check logs (zookeeper.log, kafka.log, flink.log) for details."
echo "Zookeeper PID: $ZOOKEEPER_PID"
echo "Kafka PID: $KAFKA_PID"
echo "Flink PID: $FLINK_PID"