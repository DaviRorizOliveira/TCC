import joblib
import pandas as pd
import numpy as np
from ..data_processing.feature_extraction import MatrixFeatures
import os

# Carrega o modelo e guarda as colunas esperadas
MODEL_PATH = 'data/models/best_method_model.pkl'
_model = None
_expected_columns = None

def _load_model():
    global _model, _expected_columns
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        if hasattr(_model, 'estimators_') and len(_model.estimators_) > 0:
            _expected_columns = _model.estimators_[0].feature_names_in_
        else:
            _expected_columns = getattr(_model, 'feature_names_in_', None)
        if _expected_columns is None:
            raise ValueError("Não foi possível extrair as colunas esperadas do modelo.")
    return _model, _expected_columns

def predict_best_method(A: np.ndarray):
    if not A.shape[0] == A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada.")

    # Extrai features
    features = MatrixFeatures.extract(A)
    X_raw = pd.DataFrame([features])

    # Carrega modelo
    model, expected_columns = _load_model()

    # Reordena e completa colunas faltantes com 0.0
    X = pd.DataFrame(0.0, index = X_raw.index, columns = expected_columns)
    for col in X_raw.columns:
        if col in expected_columns:
            X[col] = X_raw[col].values

    # Predição
    weights = model.predict(X)[0]

    # Detecta caso "nenhum método converge"
    if np.all(weights < 0.05):
        print("\n" + "=" * 60)
        print("           PROBLEMA DIFÍCIL DETECTADO")
        print("="*60)
        print("Nenhum dos três métodos iterativos converge bem para esta matriz.")
        print("="*60)
        return 'nenhum'

    # Normaliza
    weights = np.clip(weights, 0, None)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    methods = ['Jacobi', 'Gauss-Seidel', 'Gradiente Conjugado']
    best_idx = int(np.argmax(weights))
    best_method = ['jacobi', 'gauss_seidel', 'gradiente_conjugado'][best_idx]

    print("\n" + "=" * 60)
    print("         PREDIÇÃO DO MELHOR MÉTODO")
    print("=" * 60)
    for m, w in zip(methods, weights):
        star = " (RECOMENDADO)" if w == weights.max() else ""
        print(f"{m:<20}: {w:.1%}{star}")
    print("="*60)

    return best_method