from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from threading import Thread

# --- Basic Flask App Setup ---
# This creates the web server.
app = Flask(__name__, static_url_path='', static_folder='.')
app.config['SECRET_KEY'] = 'your-very-secret-key!' # Replace with a real secret key in production
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Serve the Frontend ---
# This route allows the server to send the index.html file to the browser.
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ==================================================================================
# == YOUR ML PIPELINE INTEGRATION GOES HERE ==
# ==================================================================================

# Define the mapping from your model's numerical output to human-readable labels.
# This is the key to fixing the '.upper()' error.
CLASS_LABELS = {
    0: 'Normal',
    1: 'DDoS',
    2: 'Port Scan',
    3: 'Infiltration' # Add other classes as needed
}


def run_ml_pipeline(filename: str) -> dict:
    """
    This function is the entry point for your entire ML pipeline.
    It takes a filename as input, performs all the necessary ML steps,
    and returns a dictionary with the results in the required format.

    Args:
        filename (str): The name of the data file to process (e.g., 'live_capture.csv').

    Returns:
        dict: A dictionary containing the analysis results. The structure of this
              dictionary MUST match what the frontend expects.
    """
    try:
        # --- STAGE 1: Data Loading & Preprocessing (Example) ---
        # Replace this with your actual data loading.
        # This dummy data now includes numerical labels, simulating a real model's output.
        data = {
            'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=1000, freq='H')),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], 1000),
            'source_ip': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(1000)],
            # --- THIS SIMULATES YOUR MODEL'S NUMERICAL OUTPUT ---
            'predicted_label': np.random.choice([0, 0, 0, 0, 1, 1, 2, 3], 1000, p=[0.9, 0.04, 0.04, 0.02])
        }
        df = pd.DataFrame(data)
        df['day_of_week'] = df['timestamp'].dt.dayofweek # 0=Monday, 6=Sunday
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date

        # --- STAGE 2: Map Numerical Predictions to String Labels (THE FIX) ---
        # This crucial step converts integer predictions (like 0, 1, 2) to string labels.
        # The error occurs when code tries to do things like .upper() on a number.
        df['anomaly_type'] = df['predicted_label'].map(CLASS_LABELS)

        # Filter out the 'Normal' traffic to focus on anomalies
        anomalies_df = df[df['anomaly_type'] != 'Normal'].copy()


        # --- STAGE 3: Result Aggregation & Formatting ---
        # This part now works correctly because it operates on the 'anomaly_type'
        # column, which contains strings, not numbers.

        # 1. Total Anomalies
        total_anomalies = len(anomalies_df)

        # 2. Traffic Volume Over Time (all traffic)
        traffic_volume = df['date'].value_counts().sort_index()

        # 3. Top Anomaly Types
        anomaly_types = anomalies_df['anomaly_type'].value_counts().to_dict()

        # 4. Protocol Distribution (all traffic)
        protocol_mix = df['protocol'].value_counts().to_dict()

        # 5. Anomaly Timeline (Heatmap Data)
        heatmap_data = anomalies_df.groupby(['day_of_week', 'hour']).size().to_dict()
        anomaly_timeline = {f"{day}-{hour}": count for (day, hour), count in heatmap_data.items()}


        # --- STAGE 4: Construct the Final Report Dictionary ---
        report = {
            "totalAnomalies": total_anomalies,
            "trafficVolume": {
                "labels": [d.strftime('%Y-%m-%d') for d in traffic_volume.index],
                "data": traffic_volume.values.tolist()
            },
            "anomalyTypes": {
                "data": anomaly_types
            },
            "protocolMix": {
                "data": protocol_mix
            },
            "anomalyTimeline": anomaly_timeline
        }
        return report

    except Exception as e:
        # If any part of your pipeline fails, return the error.
        print(f"Error in ML Pipeline: {e}")
        return {"error": str(e)}

# ==================================================================================
# == END OF ML PIPELINE INTEGRATION ==
# ==================================================================================


# --- WebSocket Communication Handler ---
def analysis_task_wrapper(sid, data):
    """
    Wrapper to run the pipeline and emit results back to the specific client.
    """
    print(f"Received report request from client {sid} for file: {data.get('filename')}")
    filename = data.get('filename', 'default_data.csv')

    report_data = run_ml_pipeline(filename)

    if "error" in report_data:
        print(f"Sending error to client {sid}: {report_data['error']}")
        socketio.emit('report_error', {'message': report_data['error']}, room=sid)
    else:
        print(f"Analysis complete. Sending report to client {sid}.")
        socketio.emit('report_ready', report_data, room=sid)

@socketio.on('request_report_data')
def handle_report_request(data):
    """
    Handles the request from the client and starts the analysis in a background thread.
    """
    sid = request.sid
    thread = Thread(target=analysis_task_wrapper, args=(sid, data))
    thread.daemon = True
    thread.start()


@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# --- Main entry point to run the server ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    print("Open your browser and go to http://127.0.0.1:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

