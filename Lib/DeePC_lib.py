import numpy as np
from qpsolvers import solve_qp

def hankel_matrix(u: np.ndarray, L: int) -> np.ndarray:
    u = np.asarray(u)
    if u.ndim != 2:
        raise ValueError("u must be a 2D array of shape (m, T).")
    m, T = u.shape
    if not (1 <= L <= T):
        raise ValueError(f"L must satisfy 1 <= L <= T (got L={L}, T={T}).")
    K = T - L + 1
    U = np.vstack([u[:, i:i+K] for i in range(L)])
    return U

class DeePC:
    def __init__(self, Up, Yp, Uf, Yf, Tini, N, Q, R, lambda_g, lambda_y, u_limit=None, y_limit=None):
        self.Up = Up
        self.Yp = Yp
        self.Uf = Uf
        self.Yf = Yf
        self.Tini = Tini
        self.N = N
        self.Q = Q
        self.R = R
        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.u_limit = u_limit
        self.y_limit = y_limit

        if u_limit is None or y_limit is None:
            self.constraint_bool = 0
        else:
            self.constraint_bool = 1

        self.T = self.Up.shape[1] + self.Tini + self.N - 1

        if self.constraint_bool:
            self.umin_col = np.tile(self.u_limit[:, 0:1], (self.N, 1))
            self.umax_col = np.tile(self.u_limit[:, 1:2], (self.N, 1))
            self.ymin_col = np.tile(self.y_limit[:, 0:1], (self.N, 1))
            self.ymax_col = np.tile(self.y_limit[:, 1:2], (self.N, 1))

        self.Q_blk = np.kron(np.eye(self.N), self.Q)
        self.R_blk = np.kron(np.eye(self.N), self.R)

        self.P = self.Yf.T @ self.Q_blk @ self.Yf + self.Uf.T @ self.R_blk @ self.Uf \
            + self.lambda_g * np.eye(self.T - self.Tini - self.N + 1) \
            + self.lambda_y * self.Yp.T @ self.Yp

        if self.constraint_bool:
            G_dense = np.vstack((self.Uf, -self.Uf, self.Yf, -self.Yf))
            self.G = G_dense
            self.h = np.vstack((self.umax_col, -self.umin_col, self.ymax_col, -self.ymin_col)).ravel()
        else:
            self.G = None
            self.h = None

        self.YfT_Q_blk = self.Yf.T @ self.Q_blk
        self.Yp_T = self.Yp.T

    def solve(self, uini, yini, y_star):
        m = uini.shape[0]
        p = yini.shape[0]

        uini_col = uini.reshape((m * self.Tini, 1), order='F')
        yini_col = yini.reshape((p * self.Tini, 1), order='F')

        y_star = np.asarray(y_star)
        
        if y_star.ndim == 1 or (y_star.ndim == 2 and y_star.shape[1] == 1):
            yr_col = np.tile(y_star.reshape(-1, 1), (self.N, 1))
        elif y_star.ndim == 2 and y_star.shape[1] == self.N:
            yr_col = y_star.reshape((p * self.N, 1), order='F')
        else:
            raise ValueError(f"y_star is invalid. Expected (p,), (p, 1) or (p, {self.N}), got {y_star.shape}")    

        q = (-self.lambda_y * self.Yp_T @ yini_col - self.YfT_Q_blk @ yr_col).ravel()

        if self.lambda_y == 0:
            A_dense = np.vstack((self.Up, self.Yp))
            A = A_dense
            b = (np.vstack((uini_col, yini_col))).ravel()
        else:
            A = self.Up
            b = (uini_col).ravel()

        try:
            g_opt = solve_qp(self.P, q, self.G, self.h, A, b, solver='quadprog')

            if g_opt is None:
                problem_status = -2
                u_opt = None
                y_opt = None
            else:
                problem_status = 1
                g_opt_col = g_opt.reshape((-1, 1))
                u_opt = self.Uf @ g_opt_col
                y_opt = self.Yf @ g_opt_col

        except Exception as e:
            print(f"An error occurred during optimization: {e}")
            problem_status = -1
            u_opt = None
            y_opt = None
            g_opt = None

        return u_opt, y_opt, g_opt, problem_status