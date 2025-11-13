import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.matrix_generator import MatrixGenerator
from src.utils.benchmark import QuickBenchmark
from src.data_processing.feature_extraction import MatrixFeatures
from src.ml_models.predict_method import predict_best_method

tamanho = 1024

def save_test_result(A, results, matrix_type = 'undefined'):
    # Salva uma linha de teste no arquivo de validação externa.
    features = MatrixFeatures.extract(A)
    
    converged = {m: r for m, r in results.items() if r['success']}
    best_method = min( converged.items(), key=lambda x: (x[1]['time'], x[1]['iterations']))[0]if converged else 'none'

    row = {
        'matrix_type': matrix_type,
        'n': A.shape[0],
        **features,
        'best_method': best_method,
        'jacobi_success': results['jacobi']['success'],
        'jacobi_iterations': results['jacobi']['iterations'] if results['jacobi']['success'] else 1000,
        'jacobi_time': results['jacobi']['time'],
        'jacobi_residual': results['jacobi']['residual'],
        'gauss_seidel_success': results['gauss_seidel']['success'],
        'gauss_seidel_iterations': results['gauss_seidel']['iterations'] if results['gauss_seidel']['success'] else 1000,
        'gauss_seidel_time': results['gauss_seidel']['time'],
        'gauss_seidel_residual': results['gauss_seidel']['residual'],
        'gradiente_conjugado_success': results['gradiente_conjugado']['success'],
        'gradiente_conjugado_iterations': results['gradiente_conjugado']['iterations'] if results['gradiente_conjugado']['success'] else 1000,
        'gradiente_conjugado_time': results['gradiente_conjugado']['time'],
        'gradiente_conjugado_residual': results['gradiente_conjugado']['residual'],
    }

    output_file = 'data/processed/test/tests.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_row = pd.DataFrame([row])
    
    if not os.path.exists(output_file):
        df_row.to_csv(output_file, index=False)
        print(f"Arquivo criado: {output_file}")
    else:
        df_row.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Linha adicionada: {output_file}")

'''
Matrizes para teste de predição:

generator.generate_random_matrix(tamanho)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 3.0)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 1.2)
generator.generate_symmetric_positive_definite(tamanho, condition_number = 50)
generator.generate_symmetric_positive_definite(tamanho, condition_number = 1000)
generator.generate_sparse_matrix(tamanho, sparsity = 0.1, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.25, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.5, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.75, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.9, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.99, ensure_convergence = True)
generator.generate_tridiagonal(tamanho, symmetric = True)
generator.generate_ill_conditioned(tamanho, condition_number = 1e5)
generator.generate_non_symmetric_dominant(tamanho, dominance_factor = 3.0)
generator.generate_indefinite(tamanho, condition_number = 100)
generator.generate_clustered_eigenvalues(tamanho, num_clusters = 2, condition_number = 100)
generator.generate_toeplitz(tamanho, alpha = 0.5)
generator.generate_laplacian_2d(m)
generator.generate_band_matrix(tamanho, bandwidth = 5)
'''

def teste():
    generator = MatrixGenerator()
    benchmark = QuickBenchmark()
    
    # m = int(np.sqrt(tamanho).round())
    A, b = generator.generate_symmetric_positive_definite(tamanho, condition_number = 500)
    
    # 1. Predição
    print('\n=======================================')
    predict_best_method(A)
    print('=======================================\n')
    
    # 2. Benchmark
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Teste de Previsão")
    
    # 3. Salvar resultado
    save_test_result(A, results)

if __name__ == "__main__":
    teste()