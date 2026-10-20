with mlflow.start_run(run_name="baseline-minilm-logreg"):
    mlflow.log_params({"data_hash": DATA_HASH, "seed": SEED, "n_train": len(fit_df), "n_val": len(val_df)})
    baseline = baseline_logreg(fit_df, val_df, test, encoder_name="minilm-l6", use_cache=False)

print(f"Baseline auf dem Testset: Accuracy {baseline['test_accuracy']:.3f}, Macro-F1 {baseline['test_macro_f1']:.3f}")
print(f"Embeddings rechnen und Regression anpassen: {baseline['train_seconds']:.0f} s, davon Regression "
      f"{baseline['fit_seconds']:.1f} s; {baseline['texts_per_second']:.0f} Texte je Sekunde")



DEFAULTS = {"learning_rate": 5e-5, "batch_size": 32}  # Empfehlung aus dem BERT-Paper, ohne Suche

with mlflow.start_run(run_name="bert-mini-default-1ep") as run:
    mlflow.log_param("data_hash", DATA_HASH)
    full_1ep = finetune(fit_df, val_df, intents, epochs=1, **DEFAULTS)
    full_1ep.update(evaluate(full_1ep["model"], full_1ep["tokenizer"], test))
    full_1ep["run_id"] = run.info.run_id

print(f"bert-mini, 1 Epoche, {DEFAULTS}: val_accuracy {full_1ep['val_accuracy']:.3f}, test_accuracy "
      f"{full_1ep['test_accuracy']:.3f}, Training {full_1ep['train_seconds']:.0f} s, "
      f"{format(full_1ep['trainable_parameters'], ',').replace(',', ' ')} trainierbare Parameter")
