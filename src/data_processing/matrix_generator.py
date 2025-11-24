import numpy as np
from scipy.linalg import toeplitz
from typing import Tuple

class MatrixGenerator:
    @staticmethod
    # Matriz aleatória densa
    def generate_random_matrix(n: int) -> Tuple[np.ndarray, np.ndarray]:
        A = np.random.uniform(-5, 5, (n, n))
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz diagonal dominante
    def generate_diagonal_dominant(n: int, dominance_factor: float = 2.0, sparse: bool = False, sparsity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        if sparse:
            A = np.random.uniform(-1, 1, (n, n))
            mask = np.random.random((n, n)) < sparsity
            A[mask] = 0
        else:
            A = np.random.uniform(-1, 1, (n, n))

        for i in range(n):
            row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            A[i, i] = dominance_factor * row_sum * np.sign(np.random.randn() or 1)

        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz simétrica positiva definida
    def generate_symmetric_positive_definite(n: int, condition_number: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        eigenvalues = np.linspace(1, condition_number, n)
        np.random.shuffle(eigenvalues)
        A = Q @ np.diag(eigenvalues) @ Q.T
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz esparsa
    def generate_sparse_matrix(n: int, sparsity: float, ensure_convergence: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        A = np.random.uniform(-5, 5, (n, n))
        mask = np.random.random((n, n)) < sparsity
        A[mask] = 0

        if ensure_convergence:
            # Torna simétrica e diagonal dominante
            A = (A + A.T) / 2
            
            # Torna diagonal dominante
            diag = np.diag(A)
            off_diag_sum = np.sum(np.abs(A - np.diag(diag)), axis = 1)
            np.fill_diagonal(A, 2.0 * off_diag_sum + 1)

        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz mal condicionada
    def generate_ill_conditioned(n: int, condition_number: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
        U, _ = np.linalg.qr(np.random.randn(n, n))
        V, _ = np.linalg.qr(np.random.randn(n, n))
        singular_values = np.logspace(0, -np.log10(condition_number), n)
        A = U @ np.diag(singular_values) @ V.T
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz tridiagonal
    def generate_tridiagonal(n: int, symmetric: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        A = np.zeros((n, n))
        if symmetric:
            for i in range(n):
                A[i, i] = 4.0
                if i > 0: A[i, i-1] = -1.0
                if i < n-1: A[i, i+1] = -1.0
        else:
            for i in range(n):
                A[i, i] = 4.0
                if i > 0: A[i, i-1] = -1.0
                if i < n-1: A[i, i+1] = -2.0
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz não simétrica e dominante
    def generate_non_symmetric_dominant(n: int, dominance_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        A = np.random.uniform(-1, 1, (n, n))
        A = A + 0.3 * np.triu(np.random.randn(n, n), k=1)
        for i in range(n):
            row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            A[i, i] = dominance_factor * row_sum + np.sign(np.random.randn() or 1)
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz indefinida
    def generate_indefinite(n: int, condition_number: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        half = n // 2
        eigenvalues = np.concatenate([
            np.linspace(1, condition_number, half),
            -np.linspace(1, condition_number, n - half)
        ])
        np.random.shuffle(eigenvalues)
        A = Q @ np.diag(eigenvalues) @ Q.T
        b = np.random.uniform(-10, 10, n)
        return A, b

    @staticmethod
    # Matriz de Toeplitz
    def generate_toeplitz(n: int, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        c = np.exp(-alpha * np.arange(n))
        A = toeplitz(c)
        A = A + np.eye(n) * 10
        b = np.random.uniform(-10, 10, n)
        return A, b