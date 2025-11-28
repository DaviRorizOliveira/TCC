import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.matrix_generator import MatrixGenerator
from src.utils.benchmark import QuickBenchmark

tamanho = 512

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
    print("\nTESTE 2: Matriz Diagonal Dominante")
    A, b = generator.generate_diagonal_dominant(n, dominance_factor = 3.0)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Diagonal Dominante (fator = 3.0)")
    
    # TESTE 3: Diagonal Dominante Fraca
    print("\nTESTE 3: Diagonal Dominante")
    A, b = generator.generate_diagonal_dominant(n, dominance_factor = 1.2)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Diagonal Dominante Fraca (fator = 1.2)")
    
    # TESTE 4: SPD Bem Condicionada
    print("\nTESTE 4: Matriz SPD")
    A, b = generator.generate_symmetric_positive_definite(n, condition_number = 50)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "SPD Bem Condicionada (κ = 50)")
    
    # TESTE 5: SPD Mal Condicionada
    print("\nTESTE 5: Matriz SPD")
    A, b = generator.generate_symmetric_positive_definite(n, condition_number = 1000)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "SPD Mal Condicionada (κ = 1000)")
    
    # TESTE 6: 25% Esparsa
    print("\nTESTE 6: Matriz 25% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.25, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "25% Esparsa (Diagonal Dominante)")
    
    # TESTE 7: 50% Esparsa
    print("\nTESTE 7: Matriz 50% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.5, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "50% Esparsa (Diagonal Dominante)")
    
    # TESTE 8: 70% Esparsa
    print("\nTESTE 8: Matriz 70% Esparsa")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.7, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "70% Esparsa (Diagonal Dominante)")
    
    # TESTE 9: Esparsa (90% zeros)
    print("\nTESTE 9: Matriz Esparsa (90% zeros)")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.9, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Esparsa 90% (Diagonal Dominante)")
    
    # TESTE 10: Esparsa (95% zeros)
    print("\nTESTE 10: Matriz Esparsa (95% zeros)")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.95, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Esparsa 95% (Diagonal Dominante)")
    
    # TESTE 11: Esparsa (99% zeros)
    print("\nTESTE 11: Matriz Esparsa (99% zeros)")
    A, b = generator.generate_sparse_matrix(n, sparsity = 0.99, ensure_convergence = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Esparsa 99% (Diagonal Dominante)")
    
    # TESTE 12: Tridiagonal
    print("\nTESTE 12: Matriz Tridiagonal Simétrica")
    A, b = generator.generate_tridiagonal(n, symmetric = True)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Tridiagonal Simétrica")

    # TESTE 13: Não Simétrica com Dominância
    print("\nTESTE 13: Matriz Não Simétrica Diagonal Dominante")
    A, b = generator.generate_non_symmetric_dominant(n, dominance_factor = 3.0)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Não Simétrica Dominante (fator = 3.0)")
    
    # TESTE 14: Indefinida (Simétrica)
    print("\nTESTE 14: Matriz Simétrica Indefinida")
    A, b = generator.generate_indefinite(n, condition_number = 100)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, "Indefinida (autovalores mistos, κ = 100)")
    
    # TESTE 15: Toeplitz
    print("\nTESTE 15: Matriz de Toeplitz")
    A, b = generator.generate_toeplitz(n, alpha = 0.5)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Toeplitz Simétrica (a = 0.5, n = {n})")

if __name__ == "__main__":
    run_beta_tests()

'''
Resultados:

TESTE 1: Matriz Completamente Aleatória
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Completamente Aleatória
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.072604        nan            
gauss_seidel              Não          ---          1.112370        nan            
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 2: Matriz Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Diagonal Dominante (fator = 3.0)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          4            0.001795        8.86e-08       
gauss_seidel              Sim          3            0.003201        9.90e-07       
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 3: Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Diagonal Dominante Fraca (fator = 1.2)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.003917        1.45e-07       
gauss_seidel              Sim          4            0.005636        3.00e-07       
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 4: Matriz SPD
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: SPD Bem Condicionada (κ = 50)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          271          0.033220        9.59e-07       
gauss_seidel              Sim          136          0.168858        9.83e-07       
gradiente_conjugado       Sim          49           0.002613        7.55e-07       
==========================================================================================

TESTE 5: Matriz SPD
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: SPD Mal Condicionada (κ = 1000)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.063939        7.14e+04       
gauss_seidel              Não          ---          1.085119        1.75e-02       
gradiente_conjugado       Sim          106          0.006148        8.64e-07       
==========================================================================================

TESTE 6: Matriz 25% Esparsa
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: 25% Esparsa (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.004345        1.07e-07       
gauss_seidel              Sim          4            0.009163        5.59e-08       
gradiente_conjugado       Sim          5            0.000410        3.33e-07       
==========================================================================================

TESTE 7: Matriz 50% Esparsa
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: 50% Esparsa (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.001534        1.74e-07       
gauss_seidel              Sim          4            0.010276        7.26e-08       
gradiente_conjugado       Sim          6            0.000766        9.23e-08       
==========================================================================================

TESTE 8: Matriz 70% Esparsa
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: 70% Esparsa (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.004024        4.65e-07       
gauss_seidel              Sim          4            0.007680        1.84e-07       
gradiente_conjugado       Sim          6            0.000568        7.00e-07       
==========================================================================================

TESTE 9: Matriz Esparsa (90% zeros)
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Esparsa 90% (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          6            0.001292        4.46e-07       
gauss_seidel              Sim          5            0.008562        2.93e-08       
gradiente_conjugado       Sim          9            0.000643        1.70e-07       
==========================================================================================

TESTE 10: Matriz Esparsa (95% zeros)
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Esparsa 95% (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          7            0.003193        3.59e-07       
gauss_seidel              Sim          5            0.007021        1.56e-07       
gradiente_conjugado       Sim          10           0.000702        4.09e-07       
==========================================================================================

TESTE 11: Matriz Esparsa (99% zeros)
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Esparsa 99% (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          11           0.001822        4.81e-07       
gauss_seidel              Sim          7            0.009917        1.64e-07       
gradiente_conjugado       Sim          30           0.004471        9.96e-07       
==========================================================================================

TESTE 12: Matriz Tridiagonal Simétrica
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Tridiagonal Simétrica
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          19           0.001837        8.62e-07       
gauss_seidel              Sim          13           0.020026        3.51e-07       
gradiente_conjugado       Sim          11           0.000944        7.02e-07       
==========================================================================================

TESTE 13: Matriz Não Simétrica Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Não Simétrica Dominante (fator = 3.0)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          4            0.001454        8.70e-08       
gauss_seidel              Sim          4            0.005685        1.01e-08       
gradiente_conjugado       Sim          5            0.000433        3.10e-07       
==========================================================================================

TESTE 14: Matriz Simétrica Indefinida
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Indefinida (autovalores mistos, κ = 100)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.062599        nan            
gauss_seidel              Não          ---          1.172096        nan            
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 15: Matriz de Toeplitz
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Toeplitz Simétrica (a = 0.5, n = 512)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          10           0.001472        4.69e-07       
gauss_seidel              Sim          6            0.011608        7.60e-07       
gradiente_conjugado       Sim          6            0.003655        2.51e-07       
==========================================================================================
'''