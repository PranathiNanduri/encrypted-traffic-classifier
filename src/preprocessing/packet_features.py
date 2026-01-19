from pathlib import Path
import csv
from scapy.all import rdpcap

PCAP_DIR = Path("data/raw/pcap")
OUT_FILE = Path("data/processed/packet_features.csv")

def extract_features(packets):
    features = []
    prev_time = None

    for pkt in packets:
        pkt_len = len(pkt)
        time = pkt.time
        iat = 0 if prev_time is None else time - prev_time
        prev_time = time

        features.append((pkt_len, iat))

    return features

def main():
    pcaps = list(PCAP_DIR.glob("*.pcap")) + list(PCAP_DIR.glob("*.pcapng"))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "packet_len", "inter_arrival_time"])

        for pcap in pcaps:
            packets = rdpcap(str(pcap))
            feats = extract_features(packets)

            for pkt_len, iat in feats:
                writer.writerow([pcap.name, pkt_len, iat])

    print("✅ Packet features extracted")

if __name__ == "__main__":
    main()
