import pandas as pd
import numpy as np
from .matrix_generator import MatrixGenerator
from .feature_extraction import MatrixFeatures
from ..utils.benchmark import QuickBenchmark
import os

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def compute_method_scores(results: dict) -> tuple[float, float, float]:
    """
    Recebe os resultados do benchmark e retorna pesos (scores) para cada método
    Quanto maior o score, melhor o método
    Retorna pesos normalizados (soma = 1.0) se pelo menos um método convergir
    Retorna (0.0, 0.0, 0.0) se nenhum convergir
    """
    methods = ['jacobi', 'gauss_seidel', 'gradiente_conjugado']
    raw_scores = []

    any_converged = False

    for m in methods:
        success = results[m]['success']
        time = results[m]['time']
        iterations = results[m]['iterations']

        if success and time > 0:
            '''
            Score alto = bom desempenho
            
            1.0 / (time + 1e-8) = Inverso do tempo (quanto menor o tempo, maior o score), 1e-8 para evitar divisão por zero
            (1.0 / (1 + 0.01 * iterations)) = Penalização leve por número de iterações (quanto mais iterações, menor o score)
            '''
            score = 1.0 / (time + 1e-8) * (1.0 / (1 + 0.01 * iterations)) # Leve penalidade por quantidade de iterações
            any_converged = True
        else:
            score = 0.0

        raw_scores.append(score)

    if not any_converged:
        return 0.0, 0.0, 0.0 # Nenhum método funciona

    # Normaliza apenas se pelo menos um convergiu
    total = sum(raw_scores)
    weights = [s / total for s in raw_scores]

    return weights[0], weights[1], weights[2]

def generate_sample(matrix_type, n, params, num_rhs: int = 10):
    try:
        gen_func = getattr(MatrixGenerator, matrix_type)
        
        A, b = gen_func(n, **params)
        n_real = n

        features = MatrixFeatures.extract(A)
        results = QuickBenchmark.run_all_methods(A, b, tol = 1e-6, max_iter = 1000)

        w_jacobi, w_gs, w_cg = compute_method_scores(results)
        
        # Detecta caso "nenhum método converge"
        all_failed = (w_jacobi == 0.0 and w_gs == 0.0 and w_cg == 0.0)

        sample = {
            'matrix_type': matrix_type,
            'n': n_real,
            **features,
            'best_method': 'nenhum' if all_failed else max([('jacobi', w_jacobi), ('gauss_seidel', w_gs), ('gradiente_conjugado', w_cg)], key = lambda x: x[1])[0],
            'weight_jacobi': w_jacobi,
            'weight_gauss_seidel': w_gs,
            'weight_gradiente_conjugado': w_cg,
            'all_methods_failed': all_failed,
        }
        
        for m, r in results.items():
            sample[f'{m}_success'] = r['success']
            sample[f'{m}_iterations'] = r['iterations']
            sample[f'{m}_time'] = r['time']
            sample[f'{m}_relative_error'] = r['relative_error']

        return sample
    except Exception as e:
        print(f"Erro ao gerar amostra {matrix_type}: {e}")
        return None

def generate_dataset(num_samples_per_type, n_range, max_workers, output_file):
    matrix_types = [
        # Matriz aleatória densa
        ('generate_random_matrix', {}),
        
        # Matriz diagonal dominante
        ('generate_diagonal_dominant', {'dominance_factor': 0.1}),
        ('generate_diagonal_dominant', {'dominance_factor': 0.2}),
        ('generate_diagonal_dominant', {'dominance_factor': 0.5}),
        ('generate_diagonal_dominant', {'dominance_factor': 0.8}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.2}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.5}),
        ('generate_diagonal_dominant', {'dominance_factor': 2.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 5.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 10.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 20.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 35.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 50.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 75.0}),
        ('generate_diagonal_dominant', {'dominance_factor': 100.0}),
        
        # Matriz diagonal dominante esparsa
        ('generate_diagonal_dominant', {'dominance_factor': 0.1, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 0.5, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.0, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.2, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 1.5, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 2.0, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 5.0, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 10.0, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 50.0, 'sparse': True, 'sparsity': 0.7}),
        ('generate_diagonal_dominant', {'dominance_factor': 100.0, 'sparse': True, 'sparsity': 0.7}),
        
        # Matriz simétrica positiva definida
        ('generate_symmetric_positive_definite', {'condition_number': 1}),
        ('generate_symmetric_positive_definite', {'condition_number': 2}),
        ('generate_symmetric_positive_definite', {'condition_number': 5}),
        ('generate_symmetric_positive_definite', {'condition_number': 10}),
        ('generate_symmetric_positive_definite', {'condition_number': 25}),
        ('generate_symmetric_positive_definite', {'condition_number': 50}),
        ('generate_symmetric_positive_definite', {'condition_number': 75}),
        ('generate_symmetric_positive_definite', {'condition_number': 100}),
        ('generate_symmetric_positive_definite', {'condition_number': 250}),
        ('generate_symmetric_positive_definite', {'condition_number': 500}),
        ('generate_symmetric_positive_definite', {'condition_number': 750}),
        ('generate_symmetric_positive_definite', {'condition_number': 1000}),
        ('generate_symmetric_positive_definite', {'condition_number': 2000}),
        
        # Matriz esparsa
        ('generate_sparse_matrix', {'sparsity': 0.1, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.25, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.33, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.5, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.7, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.75, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.9, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.95, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.98, 'ensure_convergence': True}),
        ('generate_sparse_matrix', {'sparsity': 0.99, 'ensure_convergence': True}),
        
        # Matriz tridiagonal
        ('generate_tridiagonal', {'symmetric': True}),
        ('generate_tridiagonal', {'symmetric': False}),
        
        # Matriz mal condicionada
        ('generate_ill_conditioned', {}),
        
        # Matriz não simétrica dominante
        ('generate_non_symmetric_dominant', {'dominance_factor': 0.1}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 0.2}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 0.5}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 0.8}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 1.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 1.2}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 1.5}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 2.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 5.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 10.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 20.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 35.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 50.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 75.0}),
        ('generate_non_symmetric_dominant', {'dominance_factor': 100.0}),
        
        # Matriz de Toeplitz
        ('generate_toeplitz', {'alpha': 0.1}),
        ('generate_toeplitz', {'alpha': 0.3}),
        ('generate_toeplitz', {'alpha': 0.5}),
        ('generate_toeplitz', {'alpha': 0.7}),
        ('generate_toeplitz', {'alpha': 1.0}),
        
        # Matriz indefinida
        ('generate_indefinite', {'condition_number': 1}),
        ('generate_indefinite', {'condition_number': 2}),
        ('generate_indefinite', {'condition_number': 5}),
        ('generate_indefinite', {'condition_number': 10}),
        ('generate_indefinite', {'condition_number': 25}),
        ('generate_indefinite', {'condition_number': 50}),
        ('generate_indefinite', {'condition_number': 75}),
        ('generate_indefinite', {'condition_number': 100}),
        ('generate_indefinite', {'condition_number': 250}),
        ('generate_indefinite', {'condition_number': 500}),
        ('generate_indefinite', {'condition_number': 750}),
        ('generate_indefinite', {'condition_number': 1000}),
        ('generate_indefinite', {'condition_number': 2000}),
        
        ('generate_singular_or_near_singular', {'rank_deficiency': 0, 'near_singular': True}),
        ('generate_singular_or_near_singular', {'rank_deficiency': 1, 'near_singular': True}),
        ('generate_singular_or_near_singular', {'rank_deficiency': 5, 'near_singular': True}),
        
        ('generate_singular_or_near_singular', {'rank_deficiency': 0, 'near_singular': False}),
        ('generate_singular_or_near_singular', {'rank_deficiency': 1, 'near_singular': False}),
        ('generate_singular_or_near_singular', {'rank_deficiency': 5, 'near_singular': False}),
        
        ('generate_jacobi_diverges', {}),
        
        ('generate_stieltjes', {}),
        
        ('generate_irreducibly_diagonally_dominant', {}),
        ('generate_irreducibly_diagonally_dominant', {'make_symmetric': True}),
    ]

    samples = []
    
    if n_range[0] == n_range[1]:
        n_fixed = n_range[0]
        get_n = lambda: n_fixed
    else:
        get_n = lambda: np.random.randint(*n_range)
    
    max_workers = min(max_workers, mp.cpu_count())
    
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        for mt, params in matrix_types:
            for _ in range(num_samples_per_type):
                n = get_n()
                current_params = params.copy()
                if mt == 'generate_laplacian_2d' and params.get('grid_size') == 'dynamic':
                    m = np.random.randint(5, 15)
                    current_params['grid_size'] = m
                futures.append(executor.submit(generate_sample, mt, n, current_params))

        for f in as_completed(futures):
            s = f.result()
            if s: samples.append(s)

    # Garante que o diretório de saída existe
    os.makedirs(os.path.dirname(output_file), exist_ok = True)

    # Verifica se o arquivo já existe
    file_exists = os.path.isfile(output_file)

    # Cria o DataFrame
    df_new = pd.DataFrame(samples)

    if len(df_new) == 0:
        print("Nenhuma amostra gerada nesta execução.")
        return pd.DataFrame()

    # Se o arquivo não existe, escreve com cabeçalho
    if not file_exists:
        df_new.to_csv(output_file, index = False, mode = 'w')
        print(f"Dataset criado: {output_file} ({len(df_new)} novas amostras)")
    
    else:
        # Se o arquivo existe, verifica se já tem cabeçalho
        with open(output_file, 'r', newline = '', encoding = 'utf-8') as f:
            first_line = f.readline().strip()
        
        has_header = first_line.startswith('matrix_type,n,') or 'matrix_type' in first_line

        # Verifica se o arquivo está vazio
        if has_header:
            # Adiciona novas linhas sem repetir o cabeçalho (colunas)
            df_new.to_csv(output_file, index = False, mode = 'a', header = False)
            print(f"{len(df_new)} amostras adicionadas ao dataset existente.")
        else:
            # Arquivo existe mas não tem cabeçalho, reescreve tudo com cabeçalho
            df_new.to_csv(output_file, index = False, mode = 'w')
            print(f"Arquivo sem cabeçalho detectado. Recriado com cabeçalho + {len(df_new)} amostras.")

    # Mostra o total de amostras no dataset final
    total_samples = len(pd.read_csv(output_file)) if os.path.getsize(output_file) > 0 else 0
    print(f"Dataset final: {output_file} possui {total_samples:,} amostras no total")
    
    return df_new