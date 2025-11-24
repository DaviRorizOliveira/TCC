# scripts/run_predict.py
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.matrix_generator import MatrixGenerator
from src.utils.benchmark import QuickBenchmark
from src.data_processing.feature_extraction import MatrixFeatures
from src.ml_models.predict_method import predict_best_method

tamanho = 100

def compute_real_weights(results: dict) -> tuple[float, float, float]:
    # Calcula os pesos reais com base no desempenho observado
    methods = ['jacobi', 'gauss_seidel', 'gradiente_conjugado']
    scores = []

    any_converged = False

    for m in methods:
        success = results[m]['success']
        time = results[m]['time']
        iterations = results[m]['iterations']

        if success and time > 0:
            score = 1.0 / (time + 1e-8) * (1.0 / (1 + 0.01 * iterations))
            any_converged = True
        else:
            score = 0.0
        scores.append(score)

    if not any_converged:
        return 0.0, 0.0, 0.0

    total = sum(scores)
    return scores[0] / total, scores[1] / total, scores[2] / total

def save_test_result(A: np.ndarray, b: np.ndarray, results: dict, predicted_method: str, matrix_type: str = 'undefined'):
    """
    Salva o teste gerado em um arquivo CSV
    """
    features = MatrixFeatures.extract(A)

    # Pesos reais
    w_jacobi, w_gs, w_cg = compute_real_weights(results)
    all_failed = (w_jacobi + w_gs + w_cg == 0)

    if all_failed:
        observed_best = 'nenhum'
    else:
        weights = {'jacobi': w_jacobi, 'gauss_seidel': w_gs, 'gradiente_conjugado': w_cg}
        observed_best = max(weights.items(), key=lambda x: x[1])[0]

    row = {
        'matrix_type': matrix_type,
        'n': A.shape[0],
        'sparsity': features['sparsity'],
        'symmetry': features['symmetry'],
        'condition_number_log': features['condition_number_log'],
        'diagonal_dominance_ratio': features['diagonal_dominance_ratio'],
        'diagonal_dominance_positive': features['diagonal_dominance_positive'],
        'frobenius_norm': features['frobenius_norm'],
        'trace_abs': features['trace_abs'],
        'positive_eigenvalues_ratio': features['positive_eigenvalues_ratio'],
        'has_negative_eigenvalues': features['has_negative_eigenvalues'],
        'diag_dominance_fro': features['diag_dominance_fro'],

        'best_method': observed_best,

        'weight_jacobi': w_jacobi,
        'weight_gauss_seidel': w_gs,
        'weight_gradiente_conjugado': w_cg,

        'all_methods_failed': all_failed,

        'jacobi_success': results['jacobi']['success'],
        'jacobi_iterations': results['jacobi']['iterations'] if results['jacobi']['success'] else 1000,
        'jacobi_time': results['jacobi']['time'],
        'jacobi_relative_error': results['jacobi']['relative_error'] if 'relative_error' in results['jacobi'] else 1.0,

        'gauss_seidel_success': results['gauss_seidel']['success'],
        'gauss_seidel_iterations': results['gauss_seidel']['iterations'] if results['gauss_seidel']['success'] else 1000,
        'gauss_seidel_time': results['gauss_seidel']['time'],
        'gauss_seidel_relative_error': results['gauss_seidel']['relative_error'] if 'relative_error' in results['gauss_seidel'] else 1.0,

        'gradiente_conjugado_success': results['gradiente_conjugado']['success'],
        'gradiente_conjugado_iterations': results['gradiente_conjugado']['iterations'] if results['gradiente_conjugado']['success'] else 1000,
        'gradiente_conjugado_time': results['gradiente_conjugado']['time'],
        'gradiente_conjugado_relative_error': results['gradiente_conjugado']['relative_error'] if 'relative_error' in results['gradiente_conjugado'] else 1.0,
    }

    columns_order = [
        'matrix_type', 'n', 'sparsity', 'symmetry', 'condition_number_log',
        'diagonal_dominance_ratio', 'diagonal_dominance_positive', 'frobenius_norm',
        'trace_abs', 'positive_eigenvalues_ratio', 'has_negative_eigenvalues',
        'diag_dominance_fro', 'best_method', 'weight_jacobi', 'weight_gauss_seidel',
        'weight_gradiente_conjugado', 'all_methods_failed',
        'jacobi_success', 'jacobi_iterations', 'jacobi_time', 'jacobi_relative_error',
        'gauss_seidel_success', 'gauss_seidel_iterations', 'gauss_seidel_time', 'gauss_seidel_relative_error',
        'gradiente_conjugado_success', 'gradiente_conjugado_iterations', 'gradiente_conjugado_time', 'gradiente_conjugado_relative_error'
    ]

    df_row = pd.DataFrame([row])[columns_order]

    output_file = 'data/processed/test/predict_tests.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok = True)

    if not os.path.exists(output_file):
        df_row.to_csv(output_file, index=False)
        print(f"Arquivo de testes criado: {output_file}")
    else:
        df_row.to_csv(output_file, mode='a', header=False, index = False)
        print(f"\nNova linha adicionada ao predict_tests.csv")

'''
Matrizes para teste de predição:

generator.generate_random_matrix(tamanho)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 0.1)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 0.5)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 1.0)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 1.2)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 1.5)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 2.0)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 3.0)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 5.0)
generator.generate_diagonal_dominant(tamanho, dominance_factor = 10.0)
generator.generate_symmetric_positive_definite(tamanho, condition_number = 50)
generator.generate_symmetric_positive_definite(tamanho, condition_number = 1000)
generator.generate_sparse_matrix(tamanho, sparsity = 0.1, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.25, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.5, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.75, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.9, ensure_convergence = True)
generator.generate_sparse_matrix(tamanho, sparsity = 0.95, ensure_convergence = True)
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

    print("Gerando matriz de teste")
    A, b = generator.generate_sparse_matrix(tamanho, sparsity = 0.99, ensure_convergence = True)

    print(f"Matriz gerada: {A.shape}")

    # 1. Predição do modelo
    predicted_method = predict_best_method(A)

    # 2. Benchmark real
    print("\nExecutando benchmark dos 3 métodos...\n")
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Matriz de Teste (n = {tamanho})")

    # 3. Salvar resultado real + comparação
    save_test_result(A, b, results, predicted_method, matrix_type = 'undefined')

    # 4. Comparação final
    w_j, w_gs, w_cg = compute_real_weights(results)
    print("\n" + "="*60)
    print("           RESUMO FINAL DA COMPARAÇÃO")
    print("="*60)
    print(f"Método previsto       ({predicted_method.upper()})")
    if w_j + w_gs + w_cg == 0:
        print("Método real observado (NENHUM CONVERGIU)")
    else:
        real_best = max(['jacobi', 'gauss_seidel', 'gradiente_conjugado'], key = lambda m: [w_j, w_gs, w_cg][['jacobi', 'gauss_seidel', 'gradiente_conjugado'].index(m)])
        print(f"Método real observado ({real_best.upper()}, " f"pesos: J = {w_j:.1%}, GS = {w_gs:.1%}, CG = {w_cg:.1%})")
        print("ACERTO!" if predicted_method == real_best or (predicted_method == 'nenhum' and w_j + w_gs + w_cg == 0) else "ERRO NA PREDIÇÃO")
    print("="*60)

if __name__ == "__main__":
    teste()