import numpy as np
from scipy.linalg import toeplitz
from typing import Tuple

class MatrixGenerator:
    # Gera matrizes com características controladas para testes
    
    @staticmethod
    def generate_random_matrix(n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Gera uma matriz completamente aleatória densa com valores uniformes
        A = np.random.uniform(-5, 5, (n, n))
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_diagonal_dominant(n: int, dominance_factor: float = 2.0, sparse: bool = False, sparsity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        if sparse:
            A = np.random.uniform(-1, 1, (n, n))
            mask = np.random.random((n, n)) < sparsity
            A[mask] = 0
        else:
            A = np.random.uniform(-1, 1, (n, n))
        for i in range(n):
            row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            A[i, i] = dominance_factor * row_sum * np.sign(np.random.randn())
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_symmetric_positive_definite(n: int, condition_number: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        eigenvalues = np.linspace(1, condition_number, n)
        np.random.shuffle(eigenvalues)
        A = Q @ np.diag(eigenvalues) @ Q.T
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_sparse_matrix(n: int, sparsity: float, ensure_convergence: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        A = np.random.uniform(-5, 5, (n, n))
        mask = np.random.random((n, n)) < sparsity
        A[mask] = 0
        if ensure_convergence:
            for i in range(n):
                row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
                A[i, i] = 2.0 * row_sum + 1
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_ill_conditioned(n: int, condition_number: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        singular_values = np.logspace(0, -np.log10(condition_number), n)
        A = U @ np.diag(singular_values) @ V.T
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_tridiagonal(n: int, symmetric: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        A = np.zeros((n, n))
        if symmetric:
            for i in range(n):
                A[i, i] = 4.0
                if i > 0:
                    A[i, i-1] = -1.0
                if i < n-1:
                    A[i, i+1] = -1.0
        else:
            for i in range(n):
                A[i, i] = 4.0
                if i > 0:
                    A[i, i-1] = -1.0
                if i < n-1:
                    A[i, i+1] = -2.0
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    def generate_non_symmetric_dominant(n: int, dominance_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        # Matriz não simétrica com dominância diagonal
        A = np.random.uniform(-1, 1, (n, n))
        # Adiciona assimetria: perturbação triangular superior
        A = A + 0.3 * np.triu(np.random.randn(n, n), k = 1)
        for i in range(n):
            row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            A[i, i] = dominance_factor * row_sum + np.sign(np.random.randn())
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_indefinite(n: int, condition_number: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        # Matriz simétrica indefinida (autovalores positivos e negativos)
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        half = n // 2
        eigenvalues = np.concatenate([
            np.linspace(1, condition_number, half), # Positivos
            -np.linspace(1, condition_number, n - half) # Negativos
        ])
        np.random.shuffle(eigenvalues)
        A = Q @ np.diag(eigenvalues) @ Q.T
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_clustered_eigenvalues(n: int, num_clusters: int = 2, condition_number: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        # Matriz SPD com autovalores em clusters
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        cluster_size = n // num_clusters
        eigenvalues = []
        for i in range(num_clusters):
            base = condition_number ** (i / (num_clusters - 1))  # Cluster apertado
            cluster = np.linspace(base, base * 1.1, cluster_size)
            eigenvalues.extend(cluster)
        if len(eigenvalues) < n:
            eigenvalues.append(condition_number) # Ajuste final
        eigenvalues = np.array(eigenvalues[:n])
        A = Q @ np.diag(eigenvalues) @ Q.T
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_toeplitz(n: int, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        # Matriz de Toeplitz simétrica (estrutura constante nas diagonais)
        c = np.exp(-alpha * np.arange(n))
        A = toeplitz(c)
        A = A + np.eye(n) * 10 # Garante SPD e dominância
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_laplacian_2d(m: int) -> Tuple[np.ndarray, np.ndarray]:
        # Matriz Laplaciana 2D (discretização de Poisson, esparsa e SPD). n = m^2
        n = m * m
        # Tridiagonal 1D para Laplace
        T = np.diag(4 * np.ones(m)) - np.diag(np.ones(m-1), 1) - np.diag(np.ones(m-1), -1)
        I = np.eye(m)
        # Kronecker para 2D
        A = np.kron(I, T) + np.kron(T, I)
        b = np.random.uniform(-10, 10, n)
        return A, b
    
    @staticmethod
    def generate_band_matrix(n: int, bandwidth: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        # Matriz de banda (esparsa com banda de largura 2 * bandwidth + 1)
        A = np.zeros((n, n))
        half = bandwidth // 2
        for i in range(n):
            # Diagonal Principal
            A[i, i] = 4.0
            # Bandas Laterais
            for k in range(1, half + 1):
                if i - k >= 0:
                    A[i, i - k] = -1.0
                if i + k < n:
                    A[i, i + k] = -1.0
        b = np.random.uniform(-10, 10, n)
        return A, b