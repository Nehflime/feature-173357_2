import pytest
from src.model import create_model, train_model, predict

# Datos de prueba simples
sample_data = ["El espacio es fascinante", "La religión es un tema complejo", "Los gráficos por computadora son útiles"]
sample_labels = [2, 0, 1]  # Simulando categorías: 0=alt.atheism, 1=comp.graphics, 2=sci.space

def test_model_training_and_prediction():
    model = create_model()
    trained_model = train_model(model, sample_data, sample_labels)
    predictions = predict(trained_model, sample_data)
    
    assert len(predictions) == len(sample_data)
    assert all(p in [0, 1, 2] for p in predictions)
