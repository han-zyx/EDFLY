// EdgeFlinkJob.java
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.http.sink.HttpSink;
import org.json.JSONObject;

public class EdgeFlinkJob {

    public static void main(String[] args) throws Exception {
        // Set up Flink environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment
            .createLocalEnvironment()
            .setParallelism(1);  // Single thread for edge device

        // Kafka source configuration
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers("kafka-edge:9092")
            .setTopics("lp-detections")
            .setStartingOffsets(OffsetsInitializer.earliest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

        // Create data stream from Kafka
        DataStream<String> stream = env.fromSource(
            source, WatermarkStrategy.noWatermarks(), "Kafka Source"
        );

        // Processing pipeline
        stream
            .filter(new ConfidenceFilter(0.65))  // Filter low-confidence detections
            .map(new DetectionEnricher())        // Add metadata
            .addSink(HttpSink.<String>builder()  // Send to cloud
                .setEndpointUrl("https://cloud-api/detections")
                .setEncoder(new SimpleStringEncoder<>())
                .build());

        // Execute the job
        env.execute("Edge License Plate Processing");
    }

    // Filter detections by confidence score
    private static class ConfidenceFilter implements FilterFunction<String> {
        private final double threshold;
        
        public ConfidenceFilter(double threshold) {
            this.threshold = threshold;
        }

        @Override
        public boolean filter(String value) {
            JSONObject detection = new JSONObject(value);
            return detection.getJSONArray("detections")
                .getJSONObject(0)
                .getDouble("score") > threshold;
        }
    }

    // Add metadata to detections
    private static class DetectionEnricher implements MapFunction<String, String> {
        @Override
        public String map(String value) {
            JSONObject obj = new JSONObject(value);
            obj.put("edge_processed", true);
            obj.put("cloud_received", System.currentTimeMillis());
            return obj.toString();
        }
    }
}