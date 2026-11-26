"""
generate_sample_data.py
========================
Generates small SYNTHETIC datasets that mimic the column structure of
CIC-IDS2017 and UNSW-NB15, purely so the Stage 1 pipeline can be tested
end-to-end before your team has downloaded the real datasets.

*** This is NOT real data. Replace with actual CIC-IDS2017 / UNSW-NB15 CSVs
    before training any model you intend to report results on. ***

Run this once to produce sample CSVs in ../data/, then run pipeline.py.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.RandomState(42)


def generate_cic_ids2017_sample(n_rows: int = 5000, out_path: str = "../data/sample_cic_ids2017.csv"):
    attack_types = ["BENIGN"] * 80 + ["DDoS"] * 8 + ["PortScan"] * 6 + ["Bot"] * 4 + ["Infiltration"] * 2
    labels = RNG.choice(attack_types, size=n_rows)

    df = pd.DataFrame({
        " Flow Duration": RNG.exponential(scale=100000, size=n_rows),
        " Total Fwd Packets": RNG.poisson(10, size=n_rows),
        " Total Backward Packets": RNG.poisson(8, size=n_rows),
        "Total Length of Fwd Packets": RNG.exponential(500, size=n_rows),
        " Total Length of Bwd Packets": RNG.exponential(400, size=n_rows),
        " Fwd Packet Length Max": RNG.exponential(300, size=n_rows),
        " Fwd Packet Length Min": RNG.exponential(20, size=n_rows),
        " Fwd Packet Length Mean": RNG.exponential(150, size=n_rows),
        " Fwd Packet Length Std": RNG.exponential(50, size=n_rows),
        "Flow Bytes/s": RNG.exponential(1000, size=n_rows),
        " Flow Packets/s": RNG.exponential(20, size=n_rows),
        " Flow IAT Mean": RNG.exponential(500, size=n_rows),
        " Flow IAT Std": RNG.exponential(200, size=n_rows),
        " Fwd IAT Total": RNG.exponential(1000, size=n_rows),
        " Bwd IAT Total": RNG.exponential(800, size=n_rows),
        "Protocol": RNG.choice([6, 17, 0], size=n_rows),  # TCP/UDP/other
        " Destination Port": RNG.choice([80, 443, 22, 21, 8080, 3389], size=n_rows),
        " Label": labels,
    })

    # inject some inf/nan values to simulate the real dataset's quirks
    # (zero-duration flows cause division-by-zero in Bytes/s, Packets/s cols)
    n_bad = int(n_rows * 0.01)
    bad_idx = RNG.choice(n_rows, n_bad, replace=False)
    df.loc[bad_idx, "Flow Bytes/s"] = np.inf
    n_nan = int(n_rows * 0.005)
    nan_idx = RNG.choice(n_rows, n_nan, replace=False)
    df.loc[nan_idx, " Flow IAT Mean"] = np.nan

    # a constant column, to test zero-variance dropping
    df["Fwd PSH Flags"] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic CIC-IDS2017-style sample: {out_path} ({df.shape})")
    return out_path


def generate_unsw_nb15_sample(n_rows: int = 5000, out_path: str = "../data/sample_unsw_nb15.csv"):
    is_attack = RNG.choice([0, 1], size=n_rows, p=[0.85, 0.15])
    attack_cats = np.where(
        is_attack == 0,
        "Normal",
        RNG.choice(["Exploits", "Fuzzers", "DoS", "Generic", "Reconnaissance"], size=n_rows),
    )

    df = pd.DataFrame({
        "id": np.arange(1, n_rows + 1),
        "dur": RNG.exponential(1.0, size=n_rows),
        "proto": RNG.choice(["tcp", "udp", "icmp"], size=n_rows),
        "service": RNG.choice(["http", "ftp", "dns", "-", "smtp"], size=n_rows),
        "state": RNG.choice(["FIN", "CON", "INT", "REQ"], size=n_rows),
        "spkts": RNG.poisson(12, size=n_rows),
        "dpkts": RNG.poisson(10, size=n_rows),
        "sbytes": RNG.exponential(600, size=n_rows),
        "dbytes": RNG.exponential(500, size=n_rows),
        "rate": RNG.exponential(50, size=n_rows),
        "sttl": RNG.choice([62, 254, 31], size=n_rows),
        "dttl": RNG.choice([62, 254, 31], size=n_rows),
        "sload": RNG.exponential(2000, size=n_rows),
        "dload": RNG.exponential(1800, size=n_rows),
        "sinpkt": RNG.exponential(10, size=n_rows),
        "dinpkt": RNG.exponential(8, size=n_rows),
        "label": is_attack,
        "attack_cat": attack_cats,
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic UNSW-NB15-style sample: {out_path} ({df.shape})")
    return out_path


if __name__ == "__main__":
    generate_cic_ids2017_sample()
    generate_unsw_nb15_sample()
