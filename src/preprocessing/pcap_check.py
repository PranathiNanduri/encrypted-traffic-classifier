from pathlib import Path
from scapy.all import rdpcap

BASE = Path("data/raw/pcap")

def main():
    pcaps = list(BASE.rglob("*.pcap")) + list(BASE.rglob("*.pcapng"))
    if not pcaps:
        print("❌ No capture files found under data/raw/pcap/")
        return

    for p in pcaps:
        try:
            packets = rdpcap(str(p))
            print(f"✅ {p.relative_to(BASE)} → {len(packets)} packets")
        except Exception as e:
            print(f"❌ FAILED: {p.relative_to(BASE)}  |  {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
