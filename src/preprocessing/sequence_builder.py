import numpy as np

MAX_LEN = 50  # packets per flow

def pad_sequence(seq, max_len=MAX_LEN):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))
