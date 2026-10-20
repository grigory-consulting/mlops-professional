



fig, ax = plt.subplots(figsize=(8, 3.8))
for batch_size, color in ((32, LIGHT), (64, ACCENT), (128, WARN)):
    points = [(t.params["learning_rate"], t.value) for t in study.trials if t.params["batch_size"] == batch_size]
    if points:
        ax.scatter(*zip(*points), s=55, color=color, edgecolor="white", zorder=3, label=f"Batch {batch_size}")
ax.scatter([DEFAULTS["learning_rate"]], [full_1ep["val_accuracy"]], marker="*", s=220, color=INK, zorder=4,
           label="Paper-Einstellung")
ax.set_xscale("log")
ax.set_xlabel("Lernrate (log)")
ax.set_ylabel("Accuracy (Validierung)")
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=9)
plt.show()







