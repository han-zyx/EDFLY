from kafka import KafkaProducer
import time
import json
import random

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Generate and send JSON data in real time
id = 1
while True:
    data = {
        "id": id,
        "name": f"User_{id}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    producer.send('input-topic', value=data)
    print(f"Sent: {data}")
    id += 1
    time.sleep(3)  # Send data every 3 seconds