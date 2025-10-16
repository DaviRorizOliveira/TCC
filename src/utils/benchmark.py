import time
import warnings
from typing import Dict
import numpy as np

from ..methods.iterative_methods import IterativeMethods

class QuickBenchmark:
    # Sistema de benchmark para comparar métodos iterativos
    @staticmethod
    def run_all_methods(A: np.ndarray, b: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> Dict[str, Dict]:
        results = {}
        methods = {
            'jacobi': IterativeMethods.jacobi,
            'gauss_seidel': IterativeMethods.gauss_seidel,
            'gradiente_conjugado': IterativeMethods.conjugate_gradient
        }
        for name, method in methods.items():
            try:
                start_time = time.time()
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    x, iterations, residuals = method(A, b, tol=tol, max_iter=max_iter)
                elapsed_time = time.time() - start_time
                final_residual = residuals[-1] if residuals else np.inf
                converged = final_residual < tol
                results[name] = {
                    'success': converged,
                    'iterations': iterations,
                    'time': elapsed_time,
                    'residual': final_residual,
                    'residual_history': residuals
                }
            except Exception as e:
                results[name] = {
                    'success': False,
                    'iterations': max_iter,
                    'time': 0.0,
                    'residual': np.inf,
                    'error': str(e)
                }
        return results
    
    @staticmethod
    def print_comparison(results: Dict[str, Dict], matrix_info: str = ""):
        print("=" * 90)
        if matrix_info:
            print(f"TIPO DE MATRIZ: {matrix_info}")
        print("=" * 90)
        print(f"{'Método':<25} {'Convergiu?':<12} {'Iterações':<12} {'Tempo (s)':<15} {'Resíduo Final':<15}")
        print("-" * 90)
        for method_name, data in results.items():
            status = "Sim" if data['success'] else "Não"
            iterations = data['iterations'] if data['success'] else "---"
            time_str = f"{data['time']:.6f}"
            residual = f"{data['residual']:.2e}"
            print(f"{method_name:<25} {status:<12} {str(iterations):<12} {time_str:<15} {residual:<15}")
        print("=" * 90)