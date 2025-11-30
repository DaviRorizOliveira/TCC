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
    benchmark.print_comparison(results, f"Toeplitz Simétrica (alpha = 0.5")

    # TESTE 16: Matriz Stieltjes
    print("\nTESTE 16: Matriz Stieltjes")
    A, b = generator.generate_stieltjes(n)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Stieltjes")

    # TESTE 17: Matriz Singular
    print("\nTESTE 17: Matriz Singular")
    A, b = generator.generate_singular_or_near_singular(n, near_singular = False)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Singular")
    
    # TESTE 18: Matriz Quase Singular
    print("\nTESTE 18: Matriz Quase Singular")
    A, b = generator.generate_singular_or_near_singular(n, rank_deficiency = 0.5)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Quase Singular")
    
    # TESTE 19: Matriz Irredutivelmente Diagonal Dominante
    print("\nTESTE 19: Irredutivelmente Diagonal Dominante")
    A, b = generator.generate_irreducibly_diagonally_dominant(n)
    results = benchmark.run_all_methods(A, b)
    benchmark.print_comparison(results, f"Irredutivelmente Diagonal Dominante")

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
jacobi                    Não          ---          0.061845        nan            
gauss_seidel              Não          ---          1.075276        nan            
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 2: Matriz Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Diagonal Dominante (fator = 3.0)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          4            0.003581        8.51e-08       
gauss_seidel              Sim          3            0.008235        9.75e-07       
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 3: Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Diagonal Dominante Fraca (fator = 1.2)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.001483        1.63e-07       
gauss_seidel              Sim          4            0.006753        3.07e-07       
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 4: Matriz SPD
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: SPD Bem Condicionada (κ = 50)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          312          0.020061        9.86e-07       
gauss_seidel              Sim          151          0.204248        9.99e-07       
gradiente_conjugado       Sim          50           0.004232        8.91e-07       
==========================================================================================

TESTE 5: Matriz SPD
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: SPD Mal Condicionada (κ = 1000)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.070272        4.03e+00       
gauss_seidel              Não          ---          1.121949        7.71e-03       
gradiente_conjugado       Sim          113          0.006718        9.67e-07       
==========================================================================================

TESTE 6: Matriz 25% Esparsa
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: 25% Esparsa (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.001354        1.20e-07       
gauss_seidel              Sim          4            0.007015        4.97e-08       
gradiente_conjugado       Sim          5            0.000724        6.39e-07       
==========================================================================================

TESTE 7: Matriz 50% Esparsa
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: 50% Esparsa (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.001902        1.75e-07       
gauss_seidel              Sim          4            0.006911        8.53e-08       
gradiente_conjugado       Sim          6            0.000884        7.55e-08       
==========================================================================================

TESTE 8: Matriz 70% Esparsa
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: 70% Esparsa (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.001430        4.34e-07       
gauss_seidel              Sim          4            0.008315        1.83e-07       
gradiente_conjugado       Sim          6            0.001004        5.98e-07       
==========================================================================================

TESTE 9: Matriz Esparsa (90% zeros)
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Esparsa 90% (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          6            0.001374        3.88e-07       
gauss_seidel              Sim          5            0.007290        3.31e-08       
gradiente_conjugado       Sim          8            0.000747        9.39e-07       
==========================================================================================

TESTE 10: Matriz Esparsa (95% zeros)
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Esparsa 95% (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          7            0.001499        4.91e-07       
gauss_seidel              Sim          5            0.009835        1.77e-07       
gradiente_conjugado       Sim          10           0.000871        6.80e-07       
==========================================================================================

TESTE 11: Matriz Esparsa (99% zeros)
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Esparsa 99% (Diagonal Dominante)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          11           0.001443        3.33e-07       
gauss_seidel              Sim          7            0.008848        1.07e-07       
gradiente_conjugado       Sim          30           0.003039        6.59e-07       
==========================================================================================

TESTE 12: Matriz Tridiagonal Simétrica
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Tridiagonal Simétrica
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          19           0.001883        7.89e-07       
gauss_seidel              Sim          12           0.018335        9.86e-07       
gradiente_conjugado       Sim          11           0.001041        7.40e-07       
==========================================================================================

TESTE 13: Matriz Não Simétrica Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Não Simétrica Dominante (fator = 3.0)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          4            0.001227        1.01e-07       
gauss_seidel              Sim          4            0.006989        1.04e-08       
gradiente_conjugado       Sim          5            0.001060        3.90e-07       
==========================================================================================

TESTE 14: Matriz Simétrica Indefinida
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Indefinida (autovalores mistos, κ = 100)
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.066510        nan            
gauss_seidel              Não          ---          1.079237        nan            
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 15: Matriz de Toeplitz
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Toeplitz Simétrica (alpha = 0.5
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          10           0.001238        3.73e-07       
gauss_seidel              Sim          6            0.006954        7.02e-07       
gradiente_conjugado       Sim          6            0.000552        2.48e-07       
==========================================================================================

TESTE 16: Matriz Stieltjes
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Stieltjes
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.055496        3.85e-03       
gauss_seidel              Não          ---          1.090214        1.50e-05       
gradiente_conjugado       Sim          6            0.000430        4.60e-08       
==========================================================================================

TESTE 17: Matriz Singular
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Singular
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.073331        nan            
gauss_seidel              Não          ---          1.290719        nan            
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 18: Matriz Quase Singular
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Quase Singular
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Não          ---          0.086250        nan            
gauss_seidel              Não          ---          1.345469        nan            
gradiente_conjugado       Não          ---          0.000000        nan            
==========================================================================================

TESTE 19: Irredutivelmente Diagonal Dominante
[OK] Solução exata computada (n = 512)
==========================================================================================
TIPO DE MATRIZ: Irredutivelmente Diagonal Dominante
==========================================================================================
Método                    Convergiu?   Iterações    Tempo (s)       Erro Relativo  
------------------------------------------------------------------------------------------
jacobi                    Sim          5            0.001343        3.24e-07       
gauss_seidel              Sim          4            0.004145        5.73e-07       
gradiente_conjugado       Sim          7            0.000514        7.21e-07       
==========================================================================================
'''