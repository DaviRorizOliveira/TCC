import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, r2_score, confusion_matrix, accuracy_score, top_k_accuracy_score)

def train_model(input_file = 'data/processed/train/dataset.csv', model_file = 'data/models/best_method_model.pkl'):
    print("Carregando dataset...")
    df = pd.read_csv(input_file)
    print(f"Dataset carregado: {len(df)} amostras")

    # Colunas excluidas que não serão consideradas como features
    exclude = ['matrix_type', 'best_method', 'weight_jacobi', 'weight_gauss_seidel', 'weight_gradiente_conjugado', 'all_methods_failed']
    
    # Features que serão utilizadas
    feature_cols = [c for c in df.columns  if c not in exclude and not any(c.startswith(p) for p in ['jacobi_', 'gauss_seidel_', 'gradiente_conjugado_'])]
    
    X = df[feature_cols].fillna(0)
    y_weights = df[['weight_jacobi', 'weight_gauss_seidel', 'weight_gradiente_conjugado']]
    method_names = ['jacobi', 'gauss_seidel', 'gradiente_conjugado', 'nenhum']
    y_true_labels = np.where(df['all_methods_failed'], 'nenhum', [method_names[np.argmax(row)] for row in y_weights.values])
    y = df[['weight_jacobi', 'weight_gauss_seidel', 'weight_gradiente_conjugado']]
    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(X, y_weights, y_true_labels, test_size=0.2, random_state=42, stratify=y_true_labels)

    # Treinamento do modelo
    print("Treinando MultiOutput RandomForestRegressor...\n")
    
    start_train = time.time()
    base_rf = RandomForestRegressor(n_estimators = 500, max_depth = None, random_state = 42, n_jobs = -1)
    model = MultiOutputRegressor(base_rf)
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    print(f"Tempo total de treinamento: {train_time:.3f} segundos")

    # Predição do modelo
    start_pred = time.time()
    y_pred_weights = model.predict(X_test)
    pred_time_total = time.time() - start_pred
    pred_time_per_sample = pred_time_total / len(X_test) * 1000

    print(f"Tempo total de predição: {pred_time_total:.3f} segundos")
    print(f"Tempo médio por matriz: {pred_time_per_sample:.3f} milissegundos\n")
    
    # Mean Absolute Error (MAE), erro médio absoluto dos pesos previstos (0 a 1) em relação aos pesos reais
    mae = mean_absolute_error(y_test, y_pred_weights)
    print(f"MAE: {mae:.4f}")
    
    # R^2 Score, quanto da variação dos pesos reais o modelo consegue explicar, 0.7 é bom, 0.8 é excelente
    r2 = r2_score(y_test, y_pred_weights)
    print(f"R^2: {r2:.4f}")

    y_pred_labels = np.where(np.all(y_pred_weights <= 0.01, axis=1), 'nenhum', [method_names[np.argmax(row)] for row in y_pred_weights])

    # Top-1 Accuracy (Verificação se método previsto foi o melhor real)
    top1_acc = accuracy_score(y_true_test, y_pred_labels)
    print(f"\nTop-1 Accuracy: {top1_acc:.1%}")

    # Top-2 Accuracy (Verificação se método previsto está entre os 2 melhores reais)
    y_true_idx = np.array([method_names.index(m) if m != 'nenhum' else -1 for m in y_true_test])
    valid = y_true_idx != -1
    top2_acc = top_k_accuracy_score(y_true_idx[valid], y_pred_weights[valid], k = 2, labels = range(3))
    print(f"Top-2 Accuracy: {top2_acc:.1%}")

    # Métricas detalhadas por método
    print("\nAcurácia por método:")
    for method in method_names:
        acc = accuracy_score(y_true_test[y_true_test == method], y_pred_labels[y_true_test == method])
        count = np.sum(y_true_test == method)
        print(f"  {method:22}: {acc:.1%} ({count:,} amostras)")

    # Gera a matriz de confusão
    cm = confusion_matrix(y_true_test, y_pred_labels, labels = method_names)
    plt.figure(figsize = (9, 7))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = method_names, yticklabels = method_names, cbar = False, square = True)
    plt.title(f'Matriz de Confusão')
    plt.ylabel('Método Real')
    plt.xlabel('Método Previsto')
    plt.tight_layout()
    cm_path = 'data/reports/confusion_matrix.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok = True)
    plt.savefig(cm_path, dpi = 150)
    plt.close()
    print(f"\nMatriz de confusão salva: {cm_path}")
    
    # Gera o gráfico de importância das features
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis = 0)
    idx = np.argsort(importances)[::-1][:11]
    plt.figure(figsize = (11,6))
    plt.bar(range(11), importances[idx])
    plt.xticks(range(11), [feature_cols[i] for i in idx], rotation = 45, ha = 'right')
    plt.title("Top Features (Importância Média)")
    plt.tight_layout()
    cm_path = 'data/reports/feature_importance.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok = True)
    plt.savefig(cm_path, dpi = 150)
    plt.close()
    print(f"Importância das Features salva: {cm_path}")
    
    # Salva o modelo
    os.makedirs(os.path.dirname(model_file), exist_ok = True)
    joblib.dump(model, model_file)
    print(f"\nModelo salvo: {model_file}")

    print("Treinamento concluído com sucesso!")
    return model