import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP
import time
from collections import defaultdict

# The Golden Schema - our single source of truth
GOLDEN_SCHEMA_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'src_port', 'dst_port'
]


def capture_to_file(output_filename, duration):
    """
    Captures network traffic for a set duration and saves it to a CSV file
    conforming to the Golden Schema.
    """
    connection_stats = defaultdict(lambda: {'count': 0, 'start_time': time.time(), 'serror_count': 0})

    def get_service_name(port):
        services = {80: 'http', 443: 'https', 25: 'smtp', 22: 'ssh', 21: 'ftp'}
        return services.get(port, 'private')

    all_records = []

    def process_packet_for_file(packet):
        if not packet.haslayer(IP): return

        current_time = time.time()
        src_ip, dst_ip = packet[IP].src, packet[IP].dst
        protocol_type, service, flag, src_port, dst_port = 'other', 'other', 'OTH', 0, 0
        src_bytes, dst_bytes = len(packet), 0

        if packet.haslayer(TCP):
            protocol_type, src_port, dst_port, flag = 'tcp', packet[TCP].sport, packet[TCP].dport, str(
                packet[TCP].flags)
            service = get_service_name(dst_port) or get_service_name(src_port)
        elif packet.haslayer(UDP):
            protocol_type, src_port, dst_port = 'udp', packet[UDP].sport, packet[UDP].dport
            service = get_service_name(dst_port) or get_service_name(src_port)
        elif packet.haslayer(ICMP):
            protocol_type, service = 'icmp', 'eco_i'

        host_key = dst_ip
        connection_stats[host_key]['count'] += 1
        count = connection_stats[host_key]['count']
        serror_rate = connection_stats[host_key]['serror_count'] / count if count > 0 else 0.0
        duration_val = current_time - connection_stats[host_key]['start_time']

        record = {'duration': duration_val, 'protocol_type': protocol_type, 'service': service, 'flag': flag,
                  'src_bytes': src_bytes, 'dst_bytes': dst_bytes, 'count': count, 'srv_count': count,
                  'serror_rate': serror_rate, 'same_srv_rate': 1.0, 'src_port': src_port, 'dst_port': dst_port}

        all_records.append(record)
        print(f"Captured {len(all_records)} packets...", end='\r')

    print(f"[*] Starting capture for {duration} seconds...")
    print(f"[*] Press CTRL+C to stop early.")

    try:
        sniff(prn=process_packet_for_file, timeout=duration, store=0)
        print("\n[*] Capture finished.")
    except PermissionError:
        print("\n[ERROR] Permission denied. Please run with administrator/root privileges.")
        return

    if all_records:
        df = pd.DataFrame(all_records, columns=GOLDEN_SCHEMA_COLUMNS)
        df.to_csv(output_filename, index=False)
        print(f"\n✅ Successfully saved {len(df)} packets to '{output_filename}'")
    else:
        print("No packets were captured.")
