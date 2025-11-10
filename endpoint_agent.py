import psutil
import requests
import json
import time
import random
import sys

# --- Configuration ---
# This should be the IP address of the machine running your main dashboard server.
# For testing on the same machine, '127.0.0.1' is correct.
SERVER_URL = "http://127.0.0.1:5000/ingest"
REPORT_INTERVAL = 10  # Seconds between sending data bundles


def get_system_snapshot():
    """Collects a snapshot of system metrics from the endpoint."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
        try:
            if proc.info['cpu_percent'] > 0.1 or proc.info['memory_percent'] > 0.1:
                processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # This snapshot simulates the network features our model expects,
    # basing them on the current system activity.
    snapshot = {
        'duration': random.uniform(0.0, 2.0),
        'protocol_type': random.choice(['tcp', 'udp']),
        'service': random.choice(['http', 'private', 'domain_u']),
        'flag': random.choice(['SF', 'S0', 'REJ']),
        'src_bytes': random.randint(50, 1500),
        'dst_bytes': random.randint(0, 8000),
        'count': len(processes),  # Feature: map process count to network 'count'
        'srv_count': len(processes),
        'serror_rate': 0.0,
        'same_srv_rate': 1.0,
        'src_port': random.randint(1024, 65535),
        'dst_port': random.choice([80, 443, 53]),
    }

    # If CPU load is suspiciously high, simulate a potential attack signature
    if psutil.cpu_percent() > 80.0:
        snapshot['serror_rate'] = 1.0
        snapshot['count'] = random.randint(200, 500)

    return snapshot


def run_agent():
    """The main loop for the endpoint agent."""
    print("--- 🛡️ Endpoint Security Agent ---")
    print(f"[*] Starting agent. Reporting to {SERVER_URL} every {REPORT_INTERVAL} seconds.")
    print("[*] Press CTRL+C to stop.")

    while True:
        try:
            # Collect a bundle of data to send
            data_bundle = [get_system_snapshot() for _ in range(5)]

            try:
                response = requests.post(SERVER_URL, json=data_bundle)
                if response.status_code == 200:
                    print(f"[{time.strftime('%H:%M:%S')}] Successfully reported data bundle to server.")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ❌ Error reporting to server: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ Connection Error: Could not connect to the server.")

            time.sleep(REPORT_INTERVAL)

        except KeyboardInterrupt:
            print("\n[*] Agent shutting down.");
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}");
            time.sleep(REPORT_INTERVAL)


if __name__ == "__main__":
    run_agent()

