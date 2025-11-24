import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.matrix_generator import MatrixGenerator
from src.utils.benchmark import QuickBenchmark

tamanho = 4096

def run_beta_tests():
    generator = MatrixGenerator()
    benchmark = QuickBenchmark()
    
    n = tamanho
    
    # TESTE 1: Matriz Completamente Aleatória
    print("\nTESTE 1: Matriz Completamente Aleatória")
    A, b = generator.generate_random_matrix(n)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Completamente Aleatória")
    
    # TESTE 2: Diagonal Dominante Forte
    print("\nTESTE 2: Matriz Diagonal Dominante (Forte)")
    A, b = generator.generate_diagonal_dominant(n, dominance_factor = 3.0)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Diagonal Dominante (fator = 3.0)")
    
    # TESTE 3: Diagonal Dominante Fraca
    print("\nTESTE 3: Diagonal Dominante Fraca")
    A, b = generator.generate_diagonal_dominant(n, dominance_factor = 1.2)
    results = benchmark.run_all_methods(A, b, max_iter = 1000)
    benchmark.print_comparison(results, "Diagonal Dominante Fraca (fator = 1.2)")
    
    # TESTE 4: SPD Bem Condicionada
    print("\nTESTE 4: Matriz Simétrica Positiva Definida (Bem Condicionada)")
    A, b = generator.generate_symmetric_positive_definite(n, condition_number = 50)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "SPD Bem Condicionada (κ = 50)")
    
    # TESTE 5: SPD Mal Condicionada
    print("\nTESTE 5: Matriz SPD Mal Condicionada")
    A, b = generator.generate_symmetric_positive_definite(n, condition_number = 1000)
    results = benchmark.run_all_methods(A, b, max_iter = 1000)
    benchmark.print_comparison(results, "SPD Mal Condicionada (κ = 1000)")
    
    # TESTE 6: 10% Esparsa
    print("\nTESTE 6: Matriz 10% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.1, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "10% Esparsa (Diagonal Dominante)")
    
    # TESTE 7: 25% Esparsa
    print("\nTESTE 7: Matriz 25% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.25, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "25% Esparsa (Diagonal Dominante)")
    
    # TESTE 8: 50% Esparsa
    print("\nTESTE 8: Matriz 50% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.5, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "50% Esparsa (Diagonal Dominante)")
    
    # TESTE 9: 70% Esparsa
    print("\nTESTE 9: Matriz 70% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.7, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "70% Esparsa (Diagonal Dominante)")
    
    # TESTE 10: Esparsa (90% zeros)
    print("\nTESTE 10: Matriz Esparsa (90% zeros)")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.9, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Esparsa 90% (Diagonal Dominante)")
    
    # TESTE 11: Esparsa (95% zeros)
    print("\nTESTE 11: Matriz Esparsa (95% zeros)")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.95, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Esparsa 95% (Diagonal Dominante)")
    
    # TESTE 12: Esparsa (99% zeros)
    print("\nTESTE 12: Matriz Esparsa (99% zeros)")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.99, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Esparsa 99% (Diagonal Dominante)")
    
    # TESTE 13: Tridiagonal
    print("\nTESTE 13: Matriz Tridiagonal Simétrica")
    A, b = generator.generate_tridiagonal(n, symmetric = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Tridiagonal Simétrica")

    # TESTE 14: Não Simétrica com Dominância
    print("\nTESTE 14: Matriz Não Simétrica Diagonal Dominante")
    A, b = generator.generate_non_symmetric_dominant(n, dominance_factor = 3.0)
    results = benchmark.run_all_methods(A, b, max_iter = 1000)
    benchmark.print_comparison(results, "Não Simétrica Dominante (fator = 3.0)")
    
    # TESTE 15: Indefinida (Simétrica)
    print("\nTESTE 15: Matriz Simétrica Indefinida")
    A, b = generator.generate_indefinite(n, condition_number = 100)
    results = benchmark.run_all_methods(A, b, max_iter = 500)
    benchmark.print_comparison(results, "Indefinida (autovalores mistos, κ = 100)")
    
    # TESTE 16: Toeplitz
    print("\nTESTE 16: Matriz de Toeplitz")
    A, b = generator.generate_toeplitz(n, alpha = 0.5)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Toeplitz Simétrica (a = 0.5, n = {n})")

if __name__ == "__main__":
    run_beta_tests()