from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import threading
from datetime import datetime
import random
# --- NEW: Import the email sending function from the alerting module ---
from alerting.notifier import send_email_alert

# We are using the most basic, reliable setup. NO eventlet.
app = Flask(__name__)
# In a real app, render_template should point to index.html
app.config['SECRET_KEY'] = 'test-key'
socketio = SocketIO(app)


def generate_mock_alert():
    """Simulates the final output of the Detect -> Classify -> Explain pipeline."""
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
        'severity': 'CRITICAL'
    }


def background_sniffer_and_alerter():
    """
    Simulates the background real-time monitoring thread.

    In the final system, this function loads the ML models, reads from Scapy,
    and runs the 3-stage pipeline (Detect -> Classify -> Explain).
    """
    print("[SNIFFER] Monitoring thread started. Simulating traffic analysis...")
    count = 0
    while True:
        count += 1

        # 1. Simulate Detection (happens every 1 second)
        message = f"Traffic Window #{count}: Normal traffic window processed."
        socketio.emit('heartbeat', {'message': message})

        # 2. Simulate Anomaly Alert (Randomly happens every 10 seconds)
        if count % 10 == 0:
            alert_data = generate_mock_alert()
            alert_message = f"🚨 ANOMALY: {alert_data['classification']} | REASON: {alert_data['reason']}"

            # --- STAGE 1: Push Alert to Web UI (WebSocket) ---
            socketio.emit('anomaly_detected', alert_data)
            print(f"WEB ALERT SENT: {alert_message}")

            # --- STAGE 2: Send External Alert (Email) ---
            # Using the default recipient defined in notifier.py
            send_email_alert(alert_data)

        socketio.sleep(1)  # Use socketio.sleep instead of time.sleep in Flask-SocketIO background threads


@app.route('/')
def index():
    # We will serve a different, simpler HTML file for this test.
    # In the final project, this file is dashboard/templates/index.html
    return render_template('test_index.html')


@socketio.on('connect')
def handle_connect():
    """When a browser connects, start the monitoring thread."""
    print('[SERVER] Client connected. Starting background monitoring.')
    # Start the monitoring thread only if it's not already running
    if threading.active_count() <= 1:  # Only the main thread is active
        # Renaming to match the project's function: background_sniffer_and_alerter
        threading.Thread(target=background_sniffer_and_alerter, daemon=True).start()


if __name__ == '__main__':
    print("--- 🚀 Launching SOC Server (Email Alerting Enabled) ---")
    print("--- Go to http://127.0.0.1:5000 ---")
    # Setting debug=False is crucial for multi-threaded applications like this,
    # as Flask's reloader can spawn multiple threads, breaking the background task.
    socketio.run(app, host='127.0.0.1', port=5000, debug=False)
