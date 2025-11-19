import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''
RandomForestClassifierModelo robusto === Lida com não-linearidade, missing values, classes desbalanceadas
train_test_split === Divide treino/teste com estratificação
cross_val_score === Validação cruzada -> estimativa realista de performance
classification_report === Precisão recall, F1 por classe
confusion_matrix === Visualização de erros
'''

def train_model(input_file: str = 'data/processed/train/dataset.csv', model_file: str = 'data/models/best_method_model.pkl', test_size: float = 0.2, random_state: int = 42, n_estimators: int = 300, max_depth: int = None):
    print("Carregando dataset...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Dataset não encontrado: {input_file}")

    df = pd.read_csv(input_file)
    print(f"Dataset carregado: {len(df)} amostras, {df['best_method'].nunique()} classes")

    exclude_cols = ['matrix_type', 'best_method', 'n']
    method_cols = [col for col in df.columns if any(col.startswith(p) for p in ['jacobi_', 'gauss_seidel_', 'gradiente_conjugado_'])]
    feature_cols = [col for col in df.columns if col not in exclude_cols + method_cols]

    X = df[feature_cols].fillna(0)
    y = df['best_method']

    print(f"Features: {feature_cols}")
    print(f"Classes:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)

    print(f"\nTreinando RandomForest (n_estimators={n_estimators})...")
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = random_state, n_jobs = -1, class_weight = 'balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(classification_report(y_test, y_pred, zero_division = 0))

    print("Validação Cruzada (5-fold)...")
    cv_scores = cross_val_score(model, X, y, cv = 5, scoring = 'accuracy', n_jobs = -1)
    print(f"Acurácia CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Matriz de confusão
    cm_path = 'data/reports/confusion_matrix.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.figure(figsize = (8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred, labels = model.classes_), annot = True, fmt = 'd', cmap = 'Blues', xticklabels = model.classes_, yticklabels = model.classes_)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro'); plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(cm_path, dpi = 150); plt.close()
    print(f"Matriz salva: {cm_path}")

    # Importância
    importances = model.feature_importances_
    top_k = min(10, len(importances))
    indices = np.argsort(importances)[::-1][:top_k]
    feat_path = 'data/reports/feature_importance.png'
    plt.figure(figsize = (10, 6))
    plt.title(f"Top {top_k} Features")
    plt.bar(range(top_k), importances[indices], color = 'skyblue', edgecolor = 'black')
    plt.xticks(range(top_k), [feature_cols[i] for i in indices], rotation = 45, ha = 'right')
    plt.ylabel('Importância'); plt.tight_layout()
    plt.savefig(feat_path, dpi = 150); plt.close()
    print(f"Importância salva: {feat_path}")

    # Salvar modelo
    os.makedirs(os.path.dirname(model_file), exist_ok = True)
    joblib.dump(model, model_file)
    print(f"\nModelo salvo: {model_file}")
    print(f"Classes: {list(model.classes_)}")

    return model