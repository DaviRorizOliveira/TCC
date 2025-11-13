import numpy as np
from typing import Dict

class MatrixFeatures:
    @staticmethod
    def extract(A: np.ndarray) -> Dict[str, float]:
        n = A.shape[0]
        features = {}

        # Esparsidade
        features['sparsity'] = 1 - np.count_nonzero(A) / A.size

        # Simetria (0 a 1, onde 1 é perfeitamente simétrica)
        diff_norm = np.linalg.norm(A - A.T)
        features['symmetry'] = 1 - diff_norm / (np.linalg.norm(A) + 1e-12)

        # Estimativa do número de condição (rápida e estável)
        norm1 = np.linalg.norm(A, 1)
        norminf = np.linalg.norm(A, np.inf)
        cond_est = norm1 * norminf
        features['condition_number'] = np.log10(cond_est) if cond_est > 1 else 0

        # Dominância diagonal
        diag_abs = np.abs(np.diag(A))
        off_diag_sum = np.sum(np.abs(A - np.diag(np.diag(A))), axis=1)
        dominance = diag_abs - off_diag_sum
        features['diagonal_dominance'] = np.mean(dominance)

        # Outras features úteis
        features['size'] = n
        features['fro_norm'] = np.linalg.norm(A, 'fro')

        # Espalhamento dos autovalores (Mais lento)
        try:
            eigvals = np.linalg.eigvals(A)
            abs_eig = np.abs(eigvals)
            features['eig_spread'] = np.max(abs_eig) - np.min(abs_eig) if len(abs_eig) > 0 else 0
        except:
            features['eig_spread'] = np.nan

        return features