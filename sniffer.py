import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP
import time
import os
import argparse
from collections import defaultdict

# --- The Golden Schema ---
# This is our official "Data Contract". This sniffer will produce data in this exact format.
GOLDEN_SCHEMA_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'src_port', 'dst_port'
]

# --- Global State for Feature Calculation ---
# We use this to track connections and calculate features over a 2-second window.
connection_stats = defaultdict(lambda: {
    'count': 0,
    'start_time': time.time(),
    'serror_count': 0,
    'srv_count': defaultdict(int)
})


def get_service_name(port):
    #Maps common ports to service names. A simplified version.
    services = {
        80: 'http', 443: 'https', 25: 'smtp', 22: 'ssh', 21: 'ftp',
        20: 'ftp_data', 53: 'domain_u', 123: 'ntp_u'
    }
    return services.get(port, 'private')


def process_packet(packet):
    #Processes each sniffed packet to extract features according to the Golden Schema.
    if not packet.haslayer(IP):
        return

    # --- Basic IP and Timing Info ---
    current_time = time.time()
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    # --- Initialize Defaults ---
    protocol_type, service, flag = 'other', 'other', 'OTH'
    src_port, dst_port = 0, 0
    src_bytes = len(packet)
    dst_bytes = 0  # Simplified: Scapy doesn't easily provide response bytes

    # --- Protocol-Specific Extraction ---
    if packet.haslayer(TCP):
        protocol_type = 'tcp'
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        flag = str(packet[TCP].flags)
        service = get_service_name(dst_port) or get_service_name(src_port)
    elif packet.haslayer(UDP):
        protocol_type = 'udp'
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
        service = get_service_name(dst_port) or get_service_name(src_port)
    elif packet.haslayer(ICMP):
        protocol_type = 'icmp'
        service = 'eco_i'  # Common service for ping

    # --- Connection Feature Calculation (The "KDD Cup" features) ---
    host_key = dst_ip
    service_key = (dst_ip, service)

    # Clean up old connections (older than 2 seconds)
    expired_keys = [k for k, v in connection_stats.items() if current_time - v['start_time'] > 2]
    for k in expired_keys:
        del connection_stats[k]

    # Update stats for the current connection
    connection_stats[host_key]['count'] += 1
    connection_stats[service_key]['srv_count'][service] += 1

    is_syn_error = (protocol_type == 'tcp' and flag == 'S0')
    if is_syn_error:
        connection_stats[host_key]['serror_count'] += 1

    # Calculate derived features
    count = connection_stats[host_key]['count']
    srv_count = sum(connection_stats[service_key]['srv_count'].values())
    serror_rate = connection_stats[host_key]['serror_count'] / count if count > 0 else 0.0
    # A simple same_srv_rate approximation
    same_srv_rate = connection_stats[service_key]['srv_count'][service] / count if count > 0 else 0.0
    duration = current_time - connection_stats[host_key]['start_time']

    # --- Assemble the Record ---
    record = {
        'duration': duration, 'protocol_type': protocol_type, 'service': service, 'flag': flag,
        'src_bytes': src_bytes, 'dst_bytes': dst_bytes, 'count': count, 'srv_count': srv_count,
        'serror_rate': serror_rate, 'same_srv_rate': same_srv_rate,
        'src_port': src_port, 'dst_port': dst_port
    }

    # --- Append to CSV ---
    df_record = pd.DataFrame([record], columns=GOLDEN_SCHEMA_COLUMNS)
    write_header = not os.path.exists(args.output)
    df_record.to_csv(args.output, mode='a', header=write_header, index=False)

    # Provide live feedback to the user
    print(f"Captured: {protocol_type} packet from {src_ip}:{src_port} -> {dst_ip}:{dst_port}", end='\r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A network sniffer that produces data conforming to our project's Golden Schema.")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Duration of capture in seconds.")
    parser.add_argument("-o", "--output", type=str, default="live_capture.csv", help="Name of the output CSV file.")
    args = parser.parse_args()

    print(f"[*] Starting Golden Schema capture for {args.duration} seconds...")
    print(f"[*] Saving data to '{args.output}'. Press CTRL+C to stop early.")

    try:
        sniff(prn=process_packet, timeout=args.duration, store=0)
        print("\n[*] Capture finished.")
    except PermissionError:
        print("\n[ERROR] Permission denied. Please run with administrator/root privileges.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
