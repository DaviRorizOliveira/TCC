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
        features['symmetry'] = 1 - np.linalg.norm(A - A.T) / np.linalg.norm(A)
        
        # Número de condição (log para evitar valores extremos)
        try:
            cond = np.linalg.cond(A)
            features['condition_number'] = np.log10(cond) if cond > 0 else 0
        except:
            features['condition_number'] = np.inf
        
        # Dominância diagonal (média do fator de dominância)
        diag_abs = np.abs(np.diag(A))
        off_diag_sum = np.sum(np.abs(A - np.diag(np.diag(A))), axis = 1)
        dominance = diag_abs - off_diag_sum
        features['diagonal_dominance'] = np.mean(dominance)
        
        # Outras features úteis
        features['size'] = n
        try:
            eigvals = np.linalg.eigvals(A)
            features['eig_spread'] = np.max(np.abs(eigvals)) - np.min(np.abs(eigvals))
        except:
            features['eig_spread'] = np.nan
        
        features['fro_norm'] = np.linalg.norm(A, 'fro')
        
        return features