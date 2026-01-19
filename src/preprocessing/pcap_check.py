from pathlib import Path
from scapy.all import rdpcap

PCAP_DIR = Path("data/raw/pcap")

def main():
    pcaps = list(PCAP_DIR.glob("*.pcap")) + list(PCAP_DIR.glob("*.pcapng"))

    if not pcaps:
        print("❌ No PCAP files found")
        return

    for pcap in pcaps:
        packets = rdpcap(str(pcap))
        print(f"{pcap.name} → {len(packets)} packets")

if __name__ == "__main__":
    main()
