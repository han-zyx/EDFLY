import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaProducer;

import java.util.Properties;

public class KafkaFlinkJsonIntegration {
    public static void main(String[] args) throws Exception {
        // Set up the Flink execution environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Kafka Consumer Properties
        Properties consumerProps = new Properties();
        consumerProps.setProperty("bootstrap.servers", "localhost:9092");
        consumerProps.setProperty("group.id", "flink-consumer-group");

        // Kafka Consumer
        FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>(
            "input-topic", new SimpleStringSchema(), consumerProps);

        // Create a data stream from Kafka
        env.addSource(kafkaConsumer)
            .map(json -> {
                // Process JSON data (example: print to logs)
                System.out.println("Received: " + json);
                return json;
            })
            .addSink(new FlinkKafkaProducer<>(
                "output-topic", new SimpleStringSchema(), consumerProps));

        // Execute the Flink job
        env.execute("Kafka-Flink JSON Integration");
    }
}