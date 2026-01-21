from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import numpy as np
import logging
logging.getLogger("scapy").setLevel(logging.ERROR)

from scapy.all import rdpcap, IP, TCP, UDP

ROOT = Path("data/raw/pcap")
OUT_DIR = Path("data/processed")
MAX_LEN = 50

# ✅ EXPLICIT label mapping (never relies on enumerate order)
LABEL_MAP = {"vpn": 0, "nonvpn": 1, "tor": 2}
CLASSES = ["vpn", "nonvpn", "tor"]

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

def pcap_to_flow_sequences(pcap_path: Path):
    try:
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"⚠️ Could not read {pcap_path.name}: {e}")
        return np.zeros((0, MAX_LEN), dtype=np.float32), np.zeros((0, MAX_LEN), dtype=np.float32)

    flows = defaultdict(list)
    last_time = {}  # per-flow IAT

    for pkt in packets:
        key = get_flow_key(pkt)
        if key is None:
            continue

        pkt_len = len(pkt)
        t = float(pkt.time)
        iat = t - last_time[key] if key in last_time else 0.0
        last_time[key] = t

        flows[key].append((pkt_len, iat))

    X_len, X_iat = [], []
    for seq in flows.values():
        lengths = pad([x[0] for x in seq])
        iats = pad([x[1] for x in seq])
        X_len.append(lengths)
        X_iat.append(iats)

    return np.array(X_len, dtype=np.float32), np.array(X_iat, dtype=np.float32)

def cap_classes(X_len, X_iat, y, cap_each=250, seed=42):
    rng = np.random.default_rng(seed)
    Xl_new, Xi_new, y_new = [], [], []

    for cls in sorted(np.unique(y)):
        idx = np.where(y == cls)[0]
        if len(idx) > cap_each:
            idx = rng.choice(idx, size=cap_each, replace=False)
        Xl_new.append(X_len[idx])
        Xi_new.append(X_iat[idx])
        y_new.append(y[idx])

    X_len = np.vstack(Xl_new)
    X_iat = np.vstack(Xi_new)
    y = np.concatenate(y_new)

    p = rng.permutation(len(y))
    return X_len[p], X_iat[p], y[p]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_len, all_iat, all_y = [], [], []
    per_class_flow_counts = Counter()

    for cls in CLASSES:
        cls_dir = ROOT / cls
        if not cls_dir.exists():
            print(f"Skipping missing folder: {cls_dir}")
            continue

        label = LABEL_MAP[cls]
        pcaps = list(cls_dir.glob("*.pcap")) + list(cls_dir.glob("*.pcapng"))
        print(f"{cls}: {len(pcaps)} files")

        for pcap in pcaps:
            X_len, X_iat = pcap_to_flow_sequences(pcap)
            if X_len.shape[0] == 0:
                continue

            y = np.full((X_len.shape[0],), label, dtype=np.int64)
            all_len.append(X_len)
            all_iat.append(X_iat)
            all_y.append(y)
            per_class_flow_counts[cls] += X_len.shape[0]

    if not all_y:
        print("❌ No data found.")
        return

    X_len = np.vstack(all_len)
    X_iat = np.vstack(all_iat)
    y = np.concatenate(all_y)

    print("\n✅ Flow counts per folder (raw):", dict(per_class_flow_counts))
    print("✅ Raw counts by label id:", Counter(y.tolist()))
    print("✅ Label map:", LABEL_MAP)

    # ✅ cap TRAIN ONLY, not test
    Xlen_train, Xlen_test, Xiat_train, Xiat_test, y_train, y_test = train_test_split(
        X_len, X_iat, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n✅ Raw train:", Counter(y_train.tolist()))
    print("✅ Raw test :", Counter(y_test.tolist()))

    # cap train to balanced
    Xlen_train, Xiat_train, y_train = cap_classes(Xlen_train, Xiat_train, y_train, cap_each=250)
    print("\n✅ After cap train:", Counter(y_train.tolist()))
    print("✅ Kept test unchanged:", Counter(y_test.tolist()))

    np.savez(OUT_DIR / "train.npz", X_len=Xlen_train, X_iat=Xiat_train, y=y_train)
    np.savez(OUT_DIR / "test.npz",  X_len=Xlen_test,  X_iat=Xiat_test,  y=y_test)

    print("\n✅ Saved:")
    print(" - data/processed/train.npz")
    print(" - data/processed/test.npz")

if __name__ == "__main__":
    main()
