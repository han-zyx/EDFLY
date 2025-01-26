# metrics_server.py
from prometheus_client import start_http_server, Counter, Gauge
import psutil
import time

# Define metrics
DETECTIONS = Counter('detections_total', 'Total detections processed')
LATENCY = Gauge('processing_latency_ms', 'Detection processing latency')

def start_metrics_server(port=8000):
    start_http_server(port)
    while True:
        # Collect system metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        LATENCY.set(cpu_usage)  # Example metric
        time.sleep(5)

if __name__ == "__main__":
    start_metrics_server()