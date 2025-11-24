import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_models.train_model import train_model

'''
Carregando dataset...
Dataset carregado: 8500 amostras
Treinando MultiOutput RandomForestRegressor...

Tempo total de treinamento: 9.621 segundos
Tempo total de predição: 0.651 segundos
Tempo médio por matriz: 0.383 milissegundos

MAE: 0.0543 | R^2: 0.7153

Top-1 Accuracy: 88.5%
Top-2 Accuracy: 99.9%

Acurácia por método:
  jacobi                : 89.9% (753 amostras)
  gauss_seidel          : 10.3% (29 amostras)
  gradiente_conjugado   : 85.7% (561 amostras)
  nenhum                : 96.4% (357 amostras)

Matriz de confusão salva: data/reports/confusion_matrix.png
Importância das Features salva: data/reports/feature_importance.png

Modelo salvo: data/models/best_method_model.pkl
Treinamento concluído com sucesso!
'''

if __name__ == "__main__":
    train_model()