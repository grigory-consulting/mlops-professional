"""Stufe 4: Metric-Gate. Entscheidet, ob das Modell registriert und freigegeben wird.

Eingabe:  reports/eval_metrics.json, params.yaml (Abschnitt gate)
Ausgabe:  reports/gate.json mit status
    pass    Mindestqualität erreicht und nicht schlechter als Baseline minus Toleranz
    review  Mindestqualität erreicht, aber schlechter als die Baseline: Ein Mensch entscheidet
    fail    Mindestqualität verfehlt oder kein gültiger Wert (NaN): Das Modell wird nicht registriert
Die Stufe endet immer mit Exit-Code 0: Ein fail ist ein Ergebnis, kein Absturz.
"""
import math

import pipeline_lib as lib


def quality_gate(accuracy: float, baseline_accuracy: float, min_accuracy: float, tolerance: float) -> dict:
    # Jeder Vergleich mit NaN ergibt False: ohne diese Prüfung käme ein NaN als pass durch.
    if not (math.isfinite(accuracy) and math.isfinite(baseline_accuracy)):
        status = "fail"
        reason = f"Accuracy {accuracy} oder Baseline {baseline_accuracy} ist kein gültiger Wert."
    elif accuracy < min_accuracy:
        status = "fail"
        reason = f"Accuracy {accuracy:.3f} liegt unter der Mindestqualität {min_accuracy:.3f}."
    elif accuracy < baseline_accuracy - tolerance:
        status = "review"
        reason = (f"Accuracy {accuracy:.3f} erfüllt die Mindestqualität, liegt aber mehr als {tolerance:.3f} "
                  f"unter der Baseline {baseline_accuracy:.3f}.")
    else:
        status = "pass"
        reason = f"Accuracy {accuracy:.3f} erfüllt Mindestqualität und Baseline-Regel."
    return {"status": status, "reason": reason, "accuracy": accuracy, "baseline_accuracy": baseline_accuracy,
            "min_accuracy": min_accuracy, "tolerance": tolerance}


def main() -> None:
    cfg = lib.load_params()["gate"]
    metrics = lib.read_json("reports/eval_metrics.json")
    result = quality_gate(metrics["test_accuracy"], cfg["baseline_accuracy"], cfg["min_accuracy"], cfg["tolerance"])
    lib.write_json("reports/gate.json", result)
    print(f"gate: {result['status']} ({result['reason']})")


if __name__ == "__main__":
    main()
