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
                if i > 0: A[i, i - 1] = -1.0
                if i < n-1: A[i, i + 1] = -1.0
        else:
            for i in range(n):
                A[i, i] = 4.0
                if i > 0: A[i, i - 1] = -1.0
                if i < n-1: A[i, i + 1] = -2.0
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
    # Matriz indefinida simétrica
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
    
    @staticmethod
    # Matriz onde o método de Jacobi diverge
    def generate_jacobi_diverges(n: int):
        A = np.diag(np.ones(n)) + np.diag(0.99 * np.ones(n - 1), 1)
        A = A.T @ A + np.eye(n)
        A = np.tril(np.ones((n, n)), -1) * 1.1
        np.fill_diagonal(A, 1.0)
        b = np.ones(n)
        return A, b
    
    @staticmethod
    # Matriz de Stieltjes
    def generate_stieltjes(n: int):
        A = np.random.uniform(0.1, 2.0, (n, n))
        A = np.tril(A) + np.tril(A, -1).T
        A = -np.abs(A) + np.diag(np.sum(np.abs(A), axis = 1) + np.random.uniform(1, 5, n))
        b = np.random.rand(n)
        return A, b
    
    @staticmethod
    # Matriz singular ou quase singular
    def generate_singular_or_near_singular(n: int, rank_deficiency: int = 2, near_singular: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        R, _ = np.linalg.qr(np.random.randn(n, n))
        
        if near_singular:
            singular_vals = np.logspace(0, -16, n)
            A = Q @ np.diag(singular_vals) @ R.T
            rank_deficiency = 0
        else:
            singular_vals = np.concatenate([
                np.random.uniform(1, 10, n - rank_deficiency),
                np.zeros(rank_deficiency)
            ])
            np.random.shuffle(singular_vals)
            A = Q @ np.diag(singular_vals) @ R.T
        
        x_exact = np.random.randn(n)
        b = A @ x_exact
        
        if near_singular:
            b += np.random.randn(n) * 1e-12
        
        return A, b
    
    @staticmethod
    # Matriz irredutivelmente diagonal dominante
    def generate_irreducibly_diagonally_dominant(n: int, strict: bool = True, make_symmetric: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if n < 2:
            raise ValueError("n deve ser ≥ 2 para irreducibilidade")
        
        A = np.random.uniform(-0.9, 0.9, (n, n))
        
        for i in range(n):
            A[i, (i + 1) % n] = np.random.uniform(-0.8, -0.1)
            if i < n-1:
                A[i, i + 1] = np.random.uniform(-0.7, 0.7)
        
        for i in range(n):
            off_diag_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
            if strict:
                A[i, i] = 1.01 * off_diag_sum + 0.1 + np.random.rand()
            else:
                factor = 1.01 if i < n-1 else 1.00
                A[i, i] = factor * off_diag_sum + 0.1 + np.random.rand()
        
        if make_symmetric:
            A = (A + A.T) / 2
            for i in range(n):
                off = np.sum(np.abs(A[i, :])) - abs(A[i, i])
                A[i, i] = (1.01 if strict or i < n - 1 else 1.00) * off + 0.5
        
        A = np.array(A, dtype = float)
        b = np.random.uniform(-10, 10, n)
        
        return A, b