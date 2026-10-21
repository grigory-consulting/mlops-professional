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





from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import mmd, model
from evidently.report import Report

emb_cols = [f"e{i:03d}" for i in range(ref_emb.shape[1])]


def evidently_frame(emb: np.ndarray, signals: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je Anfrage: 384 Embedding-Spalten plus die drei skalaren Signale."""
    return pd.DataFrame(emb, columns=emb_cols).assign(**{c: signals[c].to_numpy() for c in mon.SIGNALS})


idx = windows["Themenverschiebung"]
report = Report(metrics=[EmbeddingsDriftMetric("anfrage", drift_method=model()), EmbeddingsDriftMetric("anfrage", drift_method=mmd()),
                         *[ColumnDriftMetric(column, stattest="psi") for column in mon.SIGNALS]])
report.run(reference_data=evidently_frame(ref_emb, ref_sig), current_data=evidently_frame(log.emb[idx], log.signals.iloc[idx]),
           column_mapping=ColumnMapping(embeddings={"anfrage": emb_cols}, numerical_features=list(mon.SIGNALS), target=None, prediction=None))
report_path = OUT / "drift_report.html"
report.save_html(str(report_path))
for result, method in zip((m["result"] for m in report.as_dict()["metrics"][:2]), ("model (AUC)", "mmd")):
    print(f"Evidently {method}: Drift-Score {result['drift_score']:.3f}, Drift erkannt: {result['drift_detected']}")
print("Report:", report_path.relative_to(LABS_DIR))
if not building():  
    webbrowser.open(report_path.resolve().as_uri())
