import matplotlib.pyplot as plt

# Validation results for training batch size 16, from Poreddy (2026), Table 5.
# Columns: model, precision, recall, mAP@0.5, latency (ms per image, RTX 3060 Ti)
results = [
    ("YOLOv8n", 0.89, 0.83, 0.88, 2.5),
    ("YOLOv8s", 0.90, 0.85, 0.89, 4.8),
    ("YOLOv8m", 0.91, 0.82, 0.89, 11.3),
    ("YOLOv12n", 0.89, 0.81, 0.89, 3.6),
    ("YOLOv12s", 0.88, 0.82, 0.88, 7.1),
    ("YOLOv12m", 0.90, 0.79, 0.88, 14.0),
]

FRAME_BUDGET = 1000 / 30  # ms per frame at 30 FPS (33.3 ms)

# Calculated values for Table 1
print(f"{'Model':<10}{'F1':>6}{'FPS':>7}{'Budget':>9}")
for name, p, r, m, lat in results:
    f1 = 2 * p * r / (p + r)
    fps = 1000 / lat
    budget = lat / FRAME_BUDGET * 100
    print(f"{name:<10}{f1:>6.2f}{fps:>7.0f}{budget:>8.0f}%")

# Figure 1: latency vs. 30 FPS frame budget
names = [row[0] for row in results]
latency = [row[4] for row in results]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(names, latency, color="#2a6fd6", height=0.55)
ax.axvline(FRAME_BUDGET, color="gray", linestyle="--")
ax.text(FRAME_BUDGET - 0.4, -0.6, "30 FPS budget (33.3 ms)", ha="right", va="top", color="gray")
for i, lat in enumerate(latency):
    ax.text(lat + 0.5, i, f"{lat} ms ({lat / FRAME_BUDGET * 100:.0f}% of budget)", va="center")
ax.invert_yaxis()
ax.set_xlim(0, 36)
ax.set_xlabel("Per-image inference latency on RTX 3060 Ti (ms)")
fig.tight_layout()
fig.savefig("figure1.png", dpi=300)

# Figure 2: YOLOv8s accuracy
_, p, r, m, _ = results[1]
f1 = 2 * p * r / (p + r)
labels = ["Precision", "Recall", "mAP@0.5", "F1 (calculated)"]
values = [p, r, m, f1]

fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.bar(labels, values, color=["#2a6fd6"] * 3 + ["#9db8e6"], width=0.55)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score (0-1)")
fig.tight_layout()
fig.savefig("figure2.png", dpi=300)

plt.show()