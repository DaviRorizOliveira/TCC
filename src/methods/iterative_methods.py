import numpy as np
import warnings

tolerancia = 1e-6
stop = 1000

class IterativeMethods:
    # Implementação dos três métodos iterativos: Jacobi, Gauss-Seidel, Gradiente Conjugado
    
    # 1. Método de Jacobi
    # Usa apenas valores da iteração anterior (pode ser paralelizado)
    # Converge para matrizes estritamente diagonal-dominantes
    # Mais lento que Gauss-Seidel, mas mais estável em alguns casos
    @staticmethod
    def jacobi(A, b, x_exact = None, x0 = None, tol = tolerancia, max_iter = stop):
        n = b.size
        if x0 is None:
            x0 = np.zeros(n)
        x = x0.copy()
        
        D = np.diag(A) # Extrai os itens da diagonal
        
        # Verifica se há zeros na diagonal
        if np.any(D == 0):
            raise ValueError("Diagonal possui zeros")

        inv_D = 1.0 / D # Calcula o inverso da diagonal

        R = A - np.diag(D) # L + U

        for k in range(max_iter):
            # Calcula x^(k + 1) = D^(-1) * (b - (L + U)x^(k))
            x_new = inv_D * (b - R @ x)
            
            # Critério de parada do erro relativo
            if x_exact is not None:
                rel_err = np.linalg.norm(x_new - x_exact) / (np.linalg.norm(x_exact) + 1e-15)
                if rel_err < tol:
                    return x_new, k + 1

            x = x_new
        
        warnings.warn(f"Jacobi não convergiu em {max_iter} iterações")
        return x, max_iter
    
    # 2. Método de Gauss-Seidel
    # Geralmente converge mais rápido que Jacobi (2x mais rápido tipicamente)
    # Não pode ser paralelizado (depende de valores recém-calculados)
    # Converge sob as mesmas condições que Jacobi (exceto quando ultrapassa o valor máximo das iterações)
    @staticmethod
    def gauss_seidel(A, b, x_exact = None, x0 = None, tol = tolerancia, max_iter = stop):
        n = b.size
        if x0 is None:
            x0 = np.zeros(n)
        x = x0.copy()
        
        # Verifica se há zeros na diagonal
        diag = np.diag(A)
        if np.any(diag == 0):
            raise ValueError("Diagonal possui zeros")

        for k in range(max_iter):
            x_new = x.copy()
            for i in range(n):
                # A[i, :i]: pega elementos da linha i antes da diagonal (parte L)
                # x_new[:i]: usa valores já atualizados nesta iteração
                # A[i, i+1:]: pega elementos depois da diagonal (parte U)
                # x[i+1:]: usa valores antigos (ainda não atualizados)
                sum_ax = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - sum_ax) / A[i, i]

            # Critério de parada do erro relativo
            if x_exact is not None:
                rel_err = np.linalg.norm(x_new - x_exact) / (np.linalg.norm(x_exact) + 1e-15)
                if rel_err < tol:
                    return x_new, k + 1

            x = x_new

        warnings.warn(f"Gauss-Seidel não convergiu em {max_iter} iterações")
        return x_new, max_iter
    
    # 3. Método do Gradiente Conjugado
    # Muito mais rápido que Jacobi/Gauss-Seidel para matrizes SPD
    # Converge em no máximo n iterações teoricamente (na prática, menos)
    # Falha para matrizes não SPD (indefinidas, não simétricas)
    # Ideal para matrizes esparsas de grande dimensão
    @staticmethod
    def conjugate_gradient(A, b, x_exact = None, x0 = None, tol = tolerancia, max_iter = stop):
        n = b.size
        if x0 is None:
            x0 = np.zeros(n)
        x = x0.copy()
        
        r = b - A @ x
        p = r.copy()
        rs_old = np.dot(r, r)

        for k in range(max_iter):
            Ap = A @ p
            pAp = np.dot(p, Ap)
            if pAp <= 1e-14:
                raise ValueError("Matriz possivelmente não é positiva definida")

            alpha = rs_old / pAp
            x += alpha * p
            r -= alpha * Ap

            rs_new = np.dot(r, r)

            # Critério de parada do erro relativo
            if x_exact is not None:
                rel_err = np.linalg.norm(x - x_exact) / (np.linalg.norm(x_exact) + 1e-15)
                if rel_err < tol:
                    return x, k + 1

            beta = rs_new / rs_old
            p[:] = r + beta * p # atualização in-place
            rs_old = rs_new

        warnings.warn(f"Gradiente Conjugado não convergiu em {max_iter} iterações")
        return x, max_iter