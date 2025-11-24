import numpy as np
from typing import Dict

class MatrixFeatures:
    @staticmethod
    def extract(A: np.ndarray) -> Dict[str, float]:
        n = A.shape[0]
        features = {}

        # 1. Esparsidade
        features['sparsity'] = 1 - np.count_nonzero(A) / A.size

        # 2. Simetria (0 a 1)
        diff_norm = np.linalg.norm(A - A.T, 'fro')
        norm_A = np.linalg.norm(A, 'fro')
        features['symmetry'] = 1.0 if norm_A == 0 else max(0.0, 1.0 - diff_norm / (norm_A + 1e-12))

        # 3. Estimativa do número de condição
        norm1 = np.linalg.norm(A, 1)
        norminf = np.linalg.norm(A, np.inf)
        if norm1 == 0 or norminf == 0:
            cond_est = 1.0
        else:
            cond_est = norm1 * norminf / abs(np.trace(A)) if np.trace(A) != 0 else norm1 * norminf
            cond_est = min(cond_est, 1e20)
        features['condition_number_log'] = np.log10(max(cond_est, 1.0))

        # 4. Dominância diagonal (Média do quanto a diagonal domina)
        diag_abs = np.abs(np.diag(A))
        off_diag_sum = np.sum(np.abs(A - np.diag(np.diag(A))), axis=1)
        dominance_ratio = diag_abs - off_diag_sum
        features['diagonal_dominance_ratio'] = float(np.mean(dominance_ratio))
        features['diagonal_dominance_positive'] = float(np.all(dominance_ratio > 0))

        # 5. Tamanho, norma de Frobenius e Traço
        features['n'] = float(n)
        features['frobenius_norm'] = float(np.linalg.norm(A, 'fro'))
        features['trace_abs'] = float(np.abs(np.trace(A)).item())

        # 6. Proporção de autovalores positivos
        try:
            eigvals = np.linalg.eigvals(A.astype(np.float64))
            real_parts = np.real(eigvals)
            positive_ratio = np.mean(real_parts > 1e-8)
            features['positive_eigenvalues_ratio'] = float(positive_ratio)
            features['has_negative_eigenvalues'] = float(np.any(real_parts < -1e-8))
        except:
            features['positive_eigenvalues_ratio'] = 0.5
            features['has_negative_eigenvalues'] = 0.5

        # 7. Norma da diagonal vs off-diagonal
        diag_norm = np.linalg.norm(np.diag(A))
        off_norm = np.linalg.norm(A - np.diag(np.diag(A)), 'fro')
        features['diag_dominance_fro'] = diag_norm / (off_norm + 1e-12)

        return features