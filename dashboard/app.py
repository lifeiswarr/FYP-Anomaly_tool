from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading
from datetime import datetime
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'adapt-secret-key'

# Force standard threading to avoid Windows port and eventlet issues
socketio = SocketIO(app, async_mode='threading', logger=False, engineio_logger=False)

# Global flag to control the sniffing state
is_sniffing = False


def generate_mock_alert():
    """Simulates an alert with mock data for the UI."""
    classification_options = ["SYN_FLOOD", "PORT_SCAN", "DDOS_ATTACK", "BOTNET_C2"]
    reason_options = [
        "Unusually high 'count' and sharply increased 'serror_rate'.",
        "Abnormal ratio of destination ports scanned over a 5-second window.",
        "Sustained, high volume of ICMP packets from multiple sources.",
        "Traffic pattern matching known command-and-control communication profiles."
    ]
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classification': random.choice(classification_options),
        'reason': random.choice(reason_options),
        'severity': 'CRITICAL',
        'explanation': {
            'src_bytes': random.uniform(1.5, 3.5),
            'dst_bytes': random.uniform(-1.5, -0.1),
            'count': random.uniform(1.0, 2.5)
        }
    }


def background_sniffer_and_alerter():
    """Runs continuously but only emits data when is_sniffing is True."""
    global is_sniffing
    print("[SNIFFER] Background thread active.")
    count = 0

    while True:
        if is_sniffing:
            count += 1

            # 1. Heartbeat Log
            message = f"Traffic Window #{count}: Normal traffic window processed."
            socketio.emit('heartbeat', {'message': message})

            # 2. Protocol Chart Data
            chart_data = {
                'tcp': random.randint(40, 80),
                'udp': random.randint(10, 30),
                'icmp': random.randint(0, 10),
                'other': random.randint(0, 5)
            }
            socketio.emit('protocol_update', chart_data)

            # 3. Anomaly Alert (Every 5 iterations)
            if count % 5 == 0:
                alert_data = generate_mock_alert()
                socketio.emit('anomaly_detected', alert_data)
                print(f"WEB ALERT SENT: {alert_data['classification']}")

        # Standard time.sleep is required for threading mode
        time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('[SERVER] Client connected.')


@socketio.on('start_sniffing')
def handle_start():
    global is_sniffing
    is_sniffing = True
    print('[SERVER] Live monitoring STARTED.')
    socketio.emit('status_update', {'message': 'Live monitoring started...', 'level': 'success'})


@socketio.on('stop_sniffing')
def handle_stop():
    global is_sniffing
    is_sniffing = False
    print('[SERVER] Live monitoring STOPPED.')
    socketio.emit('status_update', {'message': 'Live monitoring paused.', 'level': 'warn'})


if __name__ == '__main__':
    # Initialize background thread
    threading.Thread(target=background_sniffer_and_alerter, daemon=True).start()

    print("--- 🚀 Launching A-DAPT SOC Server ---")
    print("--- Go to http://127.0.0.1:5055 ---")

    # allow_unsafe_werkzeug=True fixes the RuntimeError
    socketio.run(
        app,
        host='127.0.0.1',
        port=5055,
        debug=True,
        use_reloader=False,
        log_output=False,
        allow_unsafe_werkzeug=True
    )