import numpy as np
from Lib.DeePC_lib import DeePC
from Lib.SoftArm_lib import constant_curvature
import gymnasium as gym

def stagecost(y, y_target, Q):
    y_flat = y.flatten() if isinstance(y, np.ndarray) else np.array(y).flatten()
    y_target_flat = y_target.flatten() if isinstance(y_target, np.ndarray) else np.array(y_target).flatten()
    
    error = y_flat - y_target_flat
    cost = float(error.T @ Q @ error)
    return cost

class SoftArmEnv:
    def __init__(self, param_deepc, arm_section, y_desired, rho):
        self.Up = param_deepc[0]
        self.Yp = param_deepc[1]
        self.Uf = param_deepc[2]
        self.Yf = param_deepc[3]
        
        self.Tini = param_deepc[4]
        self.N = param_deepc[5]

        self.Q = param_deepc[6]
        self.R = param_deepc[7]
        self.lambda_g = param_deepc[8]
        self.lambda_y = param_deepc[9]
        self.u_limit = param_deepc[10]
        self.y_limit = param_deepc[11]

        if isinstance(y_desired, np.ndarray) and y_desired.ndim == 2:
            self.T = y_desired.shape[1]
        else:
            self.T = 200  

        m_ctr = int(param_deepc[2].shape[0] / self.N)
        p_ctr = int(param_deepc[3].shape[0] / self.N)
        self.uini = np.zeros((m_ctr, self.Tini))
        self.uini_copy = self.uini.copy()
        self.yini = np.zeros((p_ctr, self.Tini))
        self.yini[2, :] = -arm_section.L * 10.0
        self.yini_copy = self.yini.copy()
        
        y_des_arr = np.array(y_desired)
        if y_des_arr.ndim == 1:
            y_point = y_des_arr.reshape(-1, 1)
            req_len = self.T + self.N + 50 
            self.y_desired = np.tile(y_point, (1, req_len))
        else:
            self.y_desired = y_des_arr

        self.deepc = DeePC(self.Up, self.Yp, self.Uf, self.Yf, self.Tini, self.N, 
                           self.Q, self.R, self.lambda_g, self.lambda_y, self.u_limit, self.y_limit)

        self.m = self.Up.shape[0] // self.Tini
        self.p = self.Yp.shape[0] // self.Tini
        
        self.y_init = np.array([[0], [0], [-arm_section.L * 10.0]])
        self.y = self.y_init.copy()
        
        self.rho = rho
        self.arm_section = arm_section

        self.pulley_diameter = 10
        self.steps_per_rev = 3200
        self.circumference = self.pulley_diameter * np.pi
        self.mm_per_step = self.circumference / self.steps_per_rev

        self.l_min = 0.70 * self.arm_section.L
        self.l_max = 1.15 * self.arm_section.L
        self.arm_length_mm = 10.0 * self.arm_section.L
        self.Q_reward = 3.33 * np.eye(self.p) 

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self.p,))
        self.action_space = gym.spaces.Discrete(2)
        
        self._rng = np.random.default_rng(seed=None)
        
    def connect(self):
        pass

    def loadInitState(self, iniStateName):
        pass

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        self.uini = self.uini_copy.copy()
        self.yini = self.yini_copy.copy()
        self.k = 0
        self.useq = np.zeros((self.m, self.N))
        self.yseq = np.zeros((self.p, self.N))
        self.t = 0
        self.y = self.y_init.copy()
        initial_target = self.y_desired[:, 0]
        self.prev_error = (initial_target.flatten() - self.y.flatten()).copy()
        return self.getFeat(initial_target), {}
    
    def getFeat(self, current_target):
        error = current_target.flatten() - self.y.flatten()
        delta_error = error - self.prev_error
        self.prev_error = error.copy()
        return np.concatenate([error, delta_error])

    def step(self, action):
        if action > 0:
            y_ref_slice = self.y_desired[:, self.t : self.t + self.N]
            
            if y_ref_slice.shape[1] < self.N:
                pad_width = self.N - y_ref_slice.shape[1]
                last_col = y_ref_slice[:, -1].reshape(-1, 1)
                y_ref_slice = np.hstack([y_ref_slice, np.tile(last_col, (1, pad_width))])

            u_opt, y_opt, g_opt, status = self.deepc.solve(self.uini, self.yini, y_ref_slice)
            self.useq = u_opt.reshape((self.m, self.N), order='F')  # (m, N)
            self.yseq = y_opt.reshape((self.p, self.N), order='F')  # (p, N) 
            self.k = 0
        else:
            self.k += 1

        u_applied = np.round(self.useq[:, self.k]).astype(int)
        
        l_known_vec = self.arm_section.L - (u_applied * self.mm_per_step / 10.0)
        l_known_vec = np.clip(l_known_vec, self.l_min, self.l_max) 
        
        kappa_b, gamma_g_rad = self.arm_section.solve_forward_kinematics(l_known_vec)
        
        # theta = kappa_b * L 
        theta = kappa_b * self.arm_section.L  
        x, y, z = constant_curvature(theta, gamma_g_rad, self.arm_length_mm) 
        
        noise = self._rng.normal(0, 0.002, (3, 1))
        self.y = np.array([[x], [y], [-z]]) + noise
        
        self.uini = np.roll(self.uini, -1, axis=1)
        self.uini[:, -1] = u_applied
        
        self.yini = np.roll(self.yini, -1, axis=1)
        self.yini[:, -1] = self.y.flatten()

        current_target = self.y_desired[:, min(self.t, self.y_desired.shape[1]-1)]
        reward = stagecost(self.y, current_target, Q=self.Q_reward) * (-1) - self.rho * action
        next_state = self.getFeat(current_target)

        self.t += 1
        # terminated: when reach the terminal state (task completed or failed)
        # truncated: when reach the maximum step limit
        if self.t >= self.T:
            truncated = True  # reach the maximum step limit
        else:
            truncated = False
        
        terminated = False  # this environment has no early termination condition, only truncation

        return next_state, reward, terminated, truncated, {}
