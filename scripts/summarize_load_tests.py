"""Combine the Locust CSV files into one table and chart."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(sys.argv[1])
COUNTS = (1, 2, 4)

rows = []
for count in COUNTS:
    stats = pd.read_csv(OUT / f"containers_{count}_stats.csv")
    item = stats[stats["Name"] == "POST /predict"].iloc[0]
    rows.append({
        "containers": count,
        "requests": int(item["Request Count"]),
        "failures": int(item["Failure Count"]),
        "requests_per_s": round(float(item["Requests/s"]), 1),
        "average_ms": round(float(item["Average Response Time"])),
        "median_ms": round(float(item["Median Response Time"])),
        "p95_ms": round(float(item["95%"])),
        "p99_ms": round(float(item["99%"])),
        "max_ms": round(float(item["Max Response Time"])),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "summary.csv", index=False)
print(summary.to_string(index=False))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
panels = (("requests_per_s", "Throughput (requests/s)"), ("median_ms", "Median latency (ms)"), ("p95_ms", "95th percentile latency (ms)"))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for axis, (key, title) in zip(axes, panels):
    labels = [str(count) for count in summary["containers"]]
    bars = axis.bar(labels, summary[key], color="#2a78d6", width=0.45)
    axis.bar_label(bars, labels=[f"{value:,.0f}" for value in summary[key]], padding=3, color="#0b0b0b")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("API containers", color="#52514e")
    axis.set_yticks([])
    axis.spines["bottom"].set_color("#c3c2b7")
    axis.margins(y=0.15)
plt.tight_layout()
plt.savefig(OUT / "load_test_summary.png", dpi=150)
