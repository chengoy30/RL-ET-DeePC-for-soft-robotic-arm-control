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
            self.T = 200  # default episode length

        m_ctr = int(param_deepc[2].shape[0] / self.N)
        p_ctr = int(param_deepc[3].shape[0] / self.N)
        self.uini = np.zeros((m_ctr, self.Tini))
        self.uini_copy = self.uini.copy()
        self.yini = np.zeros((p_ctr, self.Tini))
        self.yini[2, :] = -90.0
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
        self.y = np.array([[0], [0], [-90]])
        self.rho = rho
        self.arm_section = arm_section

        # 步进电机参数：用于将步数 u 转换为绳索长度 l
        self.pulley_diameter = 10  # 滑轮直径，单位：mm
        self.steps_per_rev = 3200  # 每转步数
        self.circumference = self.pulley_diameter * np.pi  # 滑轮周长，单位：mm
        self.mm_per_step = self.circumference / self.steps_per_rev  # 每步移动距离，单位：mm

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.p,))
        self.action_space = gym.spaces.Discrete(2)
        
        self._rng = np.random.default_rng(seed=None)
        
        # 用于存储上一次的正向运动学结果，当求解失败时使用
        self.kappa_b_last = 0.0  # 默认曲率为0（直臂状态）
        self.gamma_g_rad_last = 0.0  # 默认旋转角为0
        
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
        self.y = np.array([[0], [0], [-90]])
        # 重置正向运动学的上一次值
        self.kappa_b_last = 0.0
        self.gamma_g_rad_last = 0.0
        initial_target = self.y_desired[:, 0]
        return self.getFeat(initial_target), {}
    
    def getFeat(self, current_target):
        return (current_target.flatten() - self.y.flatten())

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


        # 步进电机只能取整数步数，对控制输入进行取整
        u_applied = np.round(self.useq[:, self.k]).astype(int)
        
        # 将步进电机步数 u 转换为绳索长度 l_known_vec
        # 公式：l = L - (u * mm_per_step / 10.0)，其中 /10.0 是将 mm 转换为 cm
        l_known_vec = self.arm_section.L - (u_applied * self.mm_per_step / 10.0)
        
        # 限制绳索长度在物理可行的范围内
        # 最小长度：约为自然长度的 70%（防止过度收缩）
        # 最大长度：约为自然长度的 115%（防止过度伸长）
        l_min = 0.70 * self.arm_section.L  # 约 6.51 cm
        l_max = 1.15 * self.arm_section.L  # 约 10.70 cm
        l_known_vec = np.clip(l_known_vec, l_min, l_max)
        
        # 使用绳索长度计算正向运动学，得到弯曲曲率和旋转角
        kappa_b, gamma_g_rad = self.arm_section.solve_forward_kinematics(l_known_vec)
        
        # 如果正向运动学求解失败，使用上一次的值或默认值
        if kappa_b is None or gamma_g_rad is None:
            print(f"警告: 正向运动学求解失败 (步数 {self.t}), l = {l_known_vec}")
            kappa_b = self.kappa_b_last
            gamma_g_rad = self.gamma_g_rad_last
        else:
            # 更新上一次的值
            self.kappa_b_last = kappa_b
            self.gamma_g_rad_last = gamma_g_rad
        
        # 根据常曲率模型计算末端位置 (x, y, z)
        # theta = kappa_b * L 是弯曲角度，phi = gamma_g_rad 是旋转角
        theta = kappa_b * self.arm_section.L  # 弯曲角度
        x, y, z = constant_curvature(theta, gamma_g_rad, 10.0 * self.arm_section.L)
        
        # 将末端位置组合成输出向量 y_plant
        y_plant = np.array([[x], [y], [-z]])
        
        
        self.y = y_plant
        self.t += 1
        
        self.uini = np.column_stack([self.uini[:, 1:], self.useq[:, self.k].reshape(-1, 1)])
        self.yini = np.column_stack([self.yini[:, 1:], self.y])

        current_target = self.y_desired[:, min(self.t, self.y_desired.shape[1]-1)]
        reward = stagecost(self.y, current_target, Q = 0.33 * np.eye(self.p)) * (-1) - self.rho * action
        next_state = self.getFeat(current_target)

        # terminated: when reach the terminal state (task completed or failed)
        # truncated: when reach the maximum step limit
        if self.t >= self.T:
            truncated = True  # reach the maximum step limit
        else:
            truncated = False
        
        terminated = False  # this environment has no early termination condition, only truncation

        return next_state, reward, terminated, truncated, {}
        