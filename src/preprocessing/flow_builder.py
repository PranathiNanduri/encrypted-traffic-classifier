from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP
from pathlib import Path

PCAP_DIR = Path("data/raw/pcap")

def get_flow_key(pkt):
    if IP in pkt:
        proto = "TCP" if TCP in pkt else "UDP" if UDP in pkt else "OTHER"
        sport = pkt.sport if hasattr(pkt, "sport") else 0
        dport = pkt.dport if hasattr(pkt, "dport") else 0
        return (pkt[IP].src, pkt[IP].dst, sport, dport, proto)
    return None

def main():
    for pcap in PCAP_DIR.glob("*.pcap*"):
        packets = rdpcap(str(pcap))
        flows = defaultdict(list)

        for pkt in packets:
            key = get_flow_key(pkt)
            if key:
                flows[key].append(pkt)

        print(f"{pcap.name} → {len(flows)} flows")

if __name__ == "__main__":
    main()