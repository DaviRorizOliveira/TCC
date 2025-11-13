import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from .matrix_generator import MatrixGenerator
from .feature_extraction import MatrixFeatures
from ..utils.benchmark import QuickBenchmark
import os

def generate_sample(matrix_type, n, params):
    try:
        gen_func = getattr(MatrixGenerator, matrix_type)
        if matrix_type == 'generate_laplacian_2d':
            m = params.get('grid_size', int(np.sqrt(n).round()))
            A, b = gen_func(m)
            n = A.shape[0]
        else:
            A, b = gen_func(n, **params)

        features = MatrixFeatures.extract(A)
        results = QuickBenchmark.run_all_methods(A, b, tol = 1e-6, max_iter = 1000)

        converged = {m: r for m, r in results.items() if r['success']}
        best_method = min(converged.items(), key=lambda x: (x[1]['time'], x[1]['iterations']))[0] if converged else 'none'

        sample = {
            'matrix_type': matrix_type,
            'n': n,
            **features,
            'best_method': best_method
        }
        for m, r in results.items():
            sample[f'{m}_success'] = r['success']
            sample[f'{m}_iterations'] = r['iterations']
            sample[f'{m}_time'] = r['time']
            sample[f'{m}_residual'] = r['residual']

        return sample
    except Exception as e:
        print(f"Erro: {e}")
        return None

def generate_dataset(num_samples_per_type = 100, n_range = (100, 1000), max_workers = 4, output_file = 'data/processed/train/dataset.csv'):
    matrix_types = [
        ('generate_random_matrix', {}),
        ('generate_diagonal_dominant', {'dominance_factor': 3.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.2}),
        ('generate_symmetric_positive_definite', {'condition_number': 50}),
        ('generate_symmetric_positive_definite', {'condition_number': 1000}),
        ('generate_sparse_matrix', {'sparsity': 0.9, 'ensure_convergence': True}),
        ('generate_tridiagonal', {'symmetric': True}),
        ('generate_laplacian_2d', {'grid_size': 'dynamic'}),
    ]

    samples = []
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        for mt, params in matrix_types:
            for _ in range(num_samples_per_type):
                n = np.random.randint(*n_range)
                current_params = params.copy()
                if mt == 'generate_laplacian_2d' and params.get('grid_size') == 'dynamic':
                    m = np.random.randint(5, 15)
                    current_params['grid_size'] = m
                futures.append(executor.submit(generate_sample, mt, n, current_params))

        for f in as_completed(futures):
            s = f.result()
            if s: samples.append(s)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df = pd.DataFrame(samples)
    df.to_csv(output_file, index=False)
    print(f"Dataset gerado: {output_file} ({len(df)} amostras)")
    return df