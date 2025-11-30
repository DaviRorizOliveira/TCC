import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.dataset_generator import generate_dataset

if __name__ == "__main__":
    saida = 'data/processed/train/dataset.csv'
    tempo_inicial = time.time()
    
    # Gera problemas com tamanhos de ordem 2^k
    for tamanho in [16, 32, 64, 128, 256, 512]:
        generate_dataset(num_samples_per_type = 5, n_range = (tamanho, tamanho), max_workers = 8, output_file = saida)
    
    # Gera problemas com tamanhos fixos
    for tamanho in [10, 20, 25, 50, 75, 100, 150, 200, 300, 400, 500]:
        generate_dataset(num_samples_per_type = 5, n_range = (tamanho, tamanho), max_workers = 8, output_file = saida)
    
    # Gera problemas com tamanhos aleatórios
    generate_dataset(num_samples_per_type = 15, n_range = (10, 512), max_workers = 8, output_file = saida)
    
    tempo_final = time.time()
    duracao = tempo_final - tempo_inicial
    print(f"Tempo de execução: {duracao:.2f} segundos")
    
    '''
    Com o dataset final contendo 8500 amostras:
    
    Tempo de execução: 10408.33 segundos
    Tempo de execução: 173 minutos e 28.33 segundos
    Tempo de execução: 2 horas, 53 minutos e 28.33 segundos
    '''