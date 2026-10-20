



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





with mlflow.start_run(run_name="bert-mini-default-3ep"):
    mlflow.log_param("data_hash", DATA_HASH)
    default_3ep = cached_finetune(PREPARED / "bert-mini-default-3ep", fit_df, val_df, intents, epochs=3, **DEFAULTS)
    mlflow.set_tag("prepared", str(default_3ep["from_cache"]).lower())
    default_3ep.update(evaluate(default_3ep["model"], default_3ep["tokenizer"], test))

with mlflow.start_run(run_name="bert-mini-final-3ep") as final_run:
    mlflow.log_params({"data_hash": DATA_HASH, "hpo_parent_run_id": parent.info.run_id})
    final = cached_finetune(PREPARED / "bert-mini-final-3ep", fit_df, val_df, intents, epochs=3, **study.best_params)
    mlflow.set_tag("prepared", str(final["from_cache"]).lower())
    final.update(evaluate(final["model"], final["tokenizer"], test))

with mlflow.start_run(run_name="minilm-l6-finetuned-3ep"):
    mlflow.log_param("data_hash", DATA_HASH)
    minilm_ft = cached_finetune(PREPARED / "minilm-l6-finetuned-3ep", fit_df, val_df, intents, epochs=3,
                                learning_rate=1e-3, batch_size=64, base_model="minilm-l6")
    mlflow.set_tag("prepared", str(minilm_ft["from_cache"]).lower())
    minilm_ft.update(evaluate(minilm_ft["model"], minilm_ft["tokenizer"], test))

