# Use a base Flink image (adjusted to your Flink version 1.13.6)
FROM apache/flink:1.13.6-scala_2.11

# Set the working directory
WORKDIR /opt/flink

# Copy the entire Flink distribution (or only the necessary parts) to the container
COPY flink-1.13.6 /opt/flink

# Copy custom scripts and configurations
# COPY custom-scripts /opt/flink/custom-scripts
# COPY flink-conf.yaml /opt/flink/conf/flink-conf.yaml

# Expose necessary ports (default Flink ports)
EXPOSE 8081 6123 8082

# Set the entrypoint to start Flink cluster
ENTRYPOINT ["bin/start-cluster.sh"]
