import joblib
import numpy as np
import pandas as pd
from ..data_processing.feature_extraction import MatrixFeatures
from typing import Literal

def predict_best_method( A: np.ndarray, model_path: str = 'data/models/best_method_model.pkl') -> Literal['jacobi', 'gauss_seidel', 'gradiente_conjugado', 'none']:
    """
    Prevê o melhor método iterativo para resolver Ax = b.

    Parameters
    ----------
        A (np.ndarray): Matriz quadrada do sistema.
        model_path (str): Caminho do modelo treinado.

    Returns
    -------
        str: Nome do método recomendado.
    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")

    # Extrair features
    features = MatrixFeatures.extract(A)
    X = pd.DataFrame([features])

    # Carregar modelo
    if not joblib.load(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    model = joblib.load(model_path)

    # Predição
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = max(proba)

    print(f"Predição: {prediction.upper()}")
    print(f"Confiança: {confidence:.1%}")
    print("Probabilidades:")
    for method, prob in zip(model.classes_, proba):
        print(f"  {method:<25}: {prob:.1%}")

    return prediction