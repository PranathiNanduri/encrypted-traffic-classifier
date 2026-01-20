from pathlib import Path
from collections import defaultdict
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP

ROOT = Path("data/raw/pcap")
OUT_DIR = Path("data/processed")

MAX_LEN = 50  # packets per flow
CLASSES = ["vpn", "nonvpn", "tor"]  # keep as-is if you have all 3 folders

def get_flow_key(pkt):
    if IP not in pkt:
        return None
    proto = 6 if TCP in pkt else 17 if UDP in pkt else 0
    sport = int(pkt.sport) if hasattr(pkt, "sport") else 0
    dport = int(pkt.dport) if hasattr(pkt, "dport") else 0
    return (pkt[IP].src, pkt[IP].dst, sport, dport, proto)

def pad(seq, max_len=MAX_LEN):
    seq = seq[:max_len]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    return seq

def pcap_to_flow_sequences(pcap_path):
    packets = rdpcap(str(pcap_path))
    flows = defaultdict(list)

    prev_time = None
    for pkt in packets:
        key = get_flow_key(pkt)
        if key is None:
            continue

        pkt_len = len(pkt)
        t = float(pkt.time)
        iat = 0.0 if prev_time is None else (t - prev_time)
        prev_time = t

        flows[key].append((pkt_len, iat))

    X_len, X_iat = [], []
    for seq in flows.values():
        lengths = pad([x[0] for x in seq])
        iats = pad([x[1] for x in seq])

        X_len.append(lengths)
        X_iat.append(iats)

    return np.array(X_len, dtype=np.float32), np.array(X_iat, dtype=np.float32)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_len, all_iat, all_y = [], [], []

    for label, cls in enumerate(CLASSES):
        cls_dir = ROOT / cls
        if not cls_dir.exists():
            print(f"Skipping missing folder: {cls_dir}")
            continue

        pcaps = list(cls_dir.glob("*.pcap")) + list(cls_dir.glob("*.pcapng"))
        print(f"{cls}: {len(pcaps)} files")

        for pcap in pcaps:
            X_len, X_iat = pcap_to_flow_sequences(pcap)
            y = np.full((X_len.shape[0],), label, dtype=np.int64)

            all_len.append(X_len)
            all_iat.append(X_iat)
            all_y.append(y)

    if not all_y:
        print("❌ No data found. Check your folders and file extensions.")
        return

    X_len = np.vstack(all_len)
    X_iat = np.vstack(all_iat)
    y = np.concatenate(all_y)

    # Shuffle
    idx = np.random.permutation(len(y))
    X_len, X_iat, y = X_len[idx], X_iat[idx], y[idx]

    # Train/Test Split
    split = int(0.8 * len(y))

    np.savez(OUT_DIR / "train.npz", X_len=X_len[:split], X_iat=X_iat[:split], y=y[:split])
    np.savez(OUT_DIR / "test.npz",  X_len=X_len[split:], X_iat=X_iat[split:], y=y[split:])

    print("✅ Saved:")
    print(" - data/processed/train.npz")
    print(" - data/processed/test.npz")
    print(f"Train samples: {split}, Test samples: {len(y) - split}")

if __name__ == "__main__":
    main()
