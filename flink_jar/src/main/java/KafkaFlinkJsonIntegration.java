import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.connector.jdbc.JdbcConnectionOptions;
import org.apache.flink.connector.jdbc.JdbcExecutionOptions;
import org.apache.flink.connector.jdbc.JdbcSink;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.util.Collector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.sql.Timestamp;
import java.util.Properties;
import java.sql.DriverManager;
import java.sql.Connection;

public class KafkaFlinkJsonIntegration {
    private static final Logger LOG = LoggerFactory.getLogger(KafkaFlinkJsonIntegration.class);

    public static void main(String[] args) throws Exception {
        // Flink execution environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Kafka Properties
        Properties consumerProps = new Properties();
        consumerProps.setProperty("bootstrap.servers", "localhost:9092");
        consumerProps.setProperty("group.id", "flink-consumer-group");
        consumerProps.setProperty("auto.offset.reset", "earliest");

        // Kafka Consumer
        FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>(
            "input-topic", new SimpleStringSchema(), consumerProps);

        //database credentials
        String dbHost = "edflydb.clcy0ayas2pc.ap-south-1.rds.amazonaws.com";
        String dbName = "licenseplatesdb";
        String dbUser = "edfly_admin";
        String dbPass = "ranolika098";

        // Test DB connection
        try (Connection conn = DriverManager.getConnection(
                "jdbc:postgresql://" + dbHost + ":5432/" + dbName, dbUser, dbPass)) {
            LOG.info("Successfully connected to RDS: jdbc:postgresql://{}:5432/{}", dbHost, dbName);
        } catch (Exception e) {
            LOG.error("Failed to connect to RDS: {}", e.getMessage(), e);
            throw e;
        }

        // Process Kafka stream
        DataStream<LicensePlateDetection> stream = env.addSource(kafkaConsumer)
            .flatMap(new JsonFlatMapFunction());

        // Print stream to logs
        stream.print("Processed Data");

        // JDBC Sink to RDS (simple insert, no conflict handling)
        stream.addSink(JdbcSink.sink(
            "INSERT INTO license_plates (detection_id, detection_time, license_plate_text) VALUES (?, ?, ?)",
            (statement, detection) -> {
                try {
                    statement.setInt(1, detection.id);           // detection_id
                    statement.setTimestamp(2, detection.detectionTime);
                    statement.setString(3, detection.licensePlateText);
                    LOG.info("Prepared DB write: detection_id={}, time={}, plate={}", 
                             detection.id, detection.detectionTime, detection.licensePlateText);
                } catch (Exception e) {
                    LOG.error("Failed to prepare DB write: detection_id={}, time={}, plate={}, error={}", 
                              detection.id, detection.detectionTime, detection.licensePlateText, e.getMessage(), e);
                    throw e;
                }
            },
            JdbcExecutionOptions.builder()
                .withBatchSize(1)
                .withBatchIntervalMs(200)
                .build(),
            new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                .withUrl("jdbc:postgresql://" + dbHost + ":5432/" + dbName)
                .withDriverName("org.postgresql.Driver")
                .withUsername(dbUser)
                .withPassword(dbPass)
                .build()
        ));

        // Execute Flink job
        env.execute("Kafka-Flink License Plate Integration");
    }

    // Custom FlatMapFunction to parse JSON
    public static class JsonFlatMapFunction implements FlatMapFunction<String, LicensePlateDetection> {
        private static final Logger LOG = LoggerFactory.getLogger(JsonFlatMapFunction.class);

        @Override
        public void flatMap(String json, Collector<LicensePlateDetection> collector) throws Exception {
            ObjectMapper mapper = new ObjectMapper();
            LOG.info("Received JSON: {}", json);
            try {
                JsonNode root = mapper.readTree(json);
                JsonNode detections = root.get("detections");
                if (detections != null && detections.isArray()) {
                    for (JsonNode detection : detections) {
                        int id = detection.get("id").asInt();
                        String timeStr = detection.get("time").asText();
                        String licensePlateText = detection.get("license_plate_text").asText();
                        Timestamp detectionTime = Timestamp.valueOf(timeStr.replace("Z", "").replace("T", " "));
                        LOG.info("Parsed detection: id={}, time={}, plate={}", id, detectionTime, licensePlateText);
                        collector.collect(new LicensePlateDetection(id, detectionTime, licensePlateText));
                    }
                } else {
                    LOG.warn("No 'detections' array found in JSON: {}", json);
                }
            } catch (Exception e) {
                LOG.error("Error parsing JSON: {}", json, e);
                throw e;
            }
        }
    }

    // Data class for license plate detections
    public static class LicensePlateDetection {
        public final int id;
        public final Timestamp detectionTime;
        public final String licensePlateText;

        public LicensePlateDetection(int id, Timestamp detectionTime, String licensePlateText) {
            this.id = id;
            this.detectionTime = detectionTime;
            this.licensePlateText = licensePlateText;
        }

        @Override
        public String toString() {
            return "LicensePlateDetection{id=" + id + ", time=" + detectionTime + ", plate=" + licensePlateText + "}";
        }
    }
}