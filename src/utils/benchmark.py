import time
import warnings
from typing import Dict
import numpy as np
from ..methods.iterative_methods import IterativeMethods

class QuickBenchmark:
    @staticmethod
    def run_all_methods(A: np.ndarray, b: np.ndarray, tol: float = 1e-6, max_iter: int = 1000) -> Dict[str, Dict]:
        results = {}
        methods = {
            'jacobi': IterativeMethods.jacobi,
            'gauss_seidel': IterativeMethods.gauss_seidel,
            'gradiente_conjugado': IterativeMethods.conjugate_gradient
        }
        
        try:
            x_exact = np.linalg.solve(A, b)
            print(f"[OK] Solução exata computada (n = {A.shape[0]})")
        except np.linalg.LinAlgError:
            x_exact = None
            print("[AVISO] np.linalg.solve falhou")

        for name, method in methods.items():
            try:
                start_time = time.time()
                with warnings.catch_warnings(record = True):
                    warnings.simplefilter("always")
                    x, iterations = method(A, b, x_exact = x_exact, tol = tol, max_iter = max_iter)
                elapsed_time = time.time() - start_time

                if x_exact is not None:
                    rel_error = np.linalg.norm(x - x_exact) / (np.linalg.norm(x_exact) + 1e-15)
                    success = rel_error < tol
                else:
                    rel_error = np.nan
                    success = False

                results[name] = {
                    'success': success,
                    'iterations': iterations,
                    'time': elapsed_time,
                    'relative_error': rel_error,
                }
            except Exception as e:
                results[name] = {
                    'success': False,
                    'iterations': max_iter,
                    'time': 0.0,
                    'relative_error': np.nan,
                    'error': str(e)
                }
        return results

    @staticmethod
    def print_comparison(results: Dict[str, Dict], matrix_info: str = ""):
        print("=" * 90)
        if matrix_info:
            print(f"TIPO DE MATRIZ: {matrix_info}")
        print("=" * 90)
        print(f"{'Método':<25} {'Convergiu?':<12} {'Iterações':<12} {'Tempo (s)':<15} {'Erro Relativo':<15}")
        print("-" * 90)
        for method_name, data in results.items():
            status = "Sim" if data['success'] else "Não"
            iterations = data['iterations'] if data['success'] else "---"
            time_str = f"{data['time']:.6f}"
            residual = f"{data['relative_error']:.2e}" if data['relative_error'] is not None else "---"
            print(f"{method_name:<25} {status:<12} {str(iterations):<12} {time_str:<15} {residual:<15}")
        print("=" * 90)