with mlflow.start_run(run_name="baseline-minilm-logreg"):
    mlflow.log_params({"data_hash": DATA_HASH, "seed": SEED, "n_train": len(fit_df), "n_val": len(val_df)})
    baseline = baseline_logreg(fit_df, val_df, test, encoder_name="minilm-l6", use_cache=False)

print(f"Baseline auf dem Testset: Accuracy {baseline['test_accuracy']:.3f}, Macro-F1 {baseline['test_macro_f1']:.3f}")
print(f"Embeddings rechnen und Regression anpassen: {baseline['train_seconds']:.0f} s, davon Regression "
      f"{baseline['fit_seconds']:.1f} s; {baseline['texts_per_second']:.0f} Texte je Sekunde")
