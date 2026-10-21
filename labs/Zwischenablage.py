fig, ax = plt.subplots(figsize=(8.5, 4.2))
curves = {"Referenz": (conf_ref, LIGHT), "kein Drift": (conf[windows["kein Drift"]], MUTED),
          "Themenverschiebung": (conf[windows["Themenverschiebung"]], ACCENT), "Tippfehler": (conf[windows["Tippfehler"]], WARN)}
for name, (values, color) in curves.items():
    x = np.sort(values)
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, linewidth=2.6 if name == "Referenz" else 1.8, label=name)
ax.set_xlabel("Konfidenz (höchste Klassenwahrscheinlichkeit)")
ax.set_ylabel("Anteil der Anfragen bis zu diesem Wert")
ax.set_title("Kumulative Verteilung der Konfidenz: Referenz gegen aktuelle Fenster")
ax.legend(frameon=False, loc="upper left")
fig.savefig(OUT / "konfidenz_ecdf.png")
plt.show()
