import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..methods.iterative_methods import IterativeMethods
from .matrix_generator import MatrixGenerator
from .feature_extraction import MatrixFeatures
from ..utils.benchmark import QuickBenchmark

def generate_sample(matrix_type, n, params):
    try:
        gen_func = getattr(MatrixGenerator, matrix_type)
        A, b = gen_func(n, **params)
        
        features = MatrixFeatures.extract(A)
        results = QuickBenchmark.run_all_methods(A, b, tol = 1e-6, max_iter = 1000)
        
        # Determinar melhor método: menor tempo se convergiu, senão menor resíduo final (O que cheggou mais perto)
        converged = {m: r for m, r in results.items() if r['success']}
        if converged:
            best_method = min(converged, key = lambda m: converged[m]['time'])
        else:
            best_method = min(results, key = lambda m: results[m]['residual'])
            if results[best_method]['residual'] == np.inf:
                best_method = 'none'
        
        sample = {
            'matrix_type': matrix_type,
            'n': n,
            **features,
            'best_method': best_method
        }
        
        for method, res in results.items():
            sample[f'{method}_success'] = res['success']
            sample[f'{method}_iterations'] = res['iterations']
            sample[f'{method}_time'] = res['time']
            sample[f'{method}_residual'] = res['residual']
        
        return sample
    except Exception as e:
        print(f"Erro em {matrix_type} n = {n}: {str(e)}")
        return None

def generate_dataset(num_samples_per_type = 100, n_range = (10, 100), max_workers = 4, output_file = 'data/processed/dataset.csv'):
    matrix_types = [
        ('generate_random_matrix', {}),
        ('generate_diagonal_dominant', {'dominance_factor': 3.0}),
        ('generate_symmetric_positive_definite', {'condition_number': 50}),
        ('generate_sparse_matrix', {'sparsity': 0.5, 'ensure_convergence': True}),
        ('generate_ill_conditioned', {'condition_number': 1e5}),
        ('generate_tridiagonal', {'symmetric': True}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 2.0}),
        ('generate_indefinite', {'condition_number': 100}),
        ('generate_clustered_eigenvalues', {'num_clusters': 2, 'condition_number': 100}),
        ('generate_toeplitz', {'alpha': 0.5}),
        ('generate_laplacian_2d', {}),
        ('generate_band_matrix', {'bandwidth': 5})
    ]
    
    samples = []
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        for matrix_type, params in matrix_types:
            for _ in range(num_samples_per_type):
                n = np.random.randint(*n_range)
                futures.append(executor.submit(generate_sample, matrix_type, n, params))
        
        for future in as_completed(futures):
            sample = future.result()
            if sample:
                samples.append(sample)
    
    df = pd.DataFrame(samples)
    df.to_csv(output_file, index = False)
    print(f"Dataset salvo em {output_file} com {len(df)} amostras.")
    return df