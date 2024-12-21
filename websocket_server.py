from flask import Flask
from flask_socketio import SocketIO
from kafka import KafkaConsumer

app = Flask(__name__)
socketio = SocketIO(app)
consumer = KafkaConsumer('yolo-detections', bootstrap_servers='localhost:9092')

@socketio.on('connect')
def handle_connect():
    for message in consumer:
        socketio.emit('detection', message.value.decode('utf-8'))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
