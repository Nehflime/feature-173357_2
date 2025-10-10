import pytest
from src.metrics import evaluate_metrics

# Simulación de etiquetas verdaderas y predichas
y_true = [0, 1, 2, 2, 1]
y_pred = [0, 1, 2, 1, 1]

def test_evaluate_metrics():
    precision, recall, f1 = evaluate_metrics(y_true, y_pred)

    # Validar que las métricas estén en el rango válido
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0

    # Validar que la precisión sea razonable
    assert precision > 0.5
