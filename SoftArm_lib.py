import numpy as np
from scipy.optimize import fsolve, minimize, Bounds
import time

class SoftArmSection:
    def __init__(self, n: int, L: float, d: float, Kb: float, Kc: float):
        self.n = n
        self.L = L
        self.d = d
        self.Kb = Kb
        self.Kc = Kc

        self.i_py = np.arange(n)
        self.cable_angles = (2 * np.pi * self.i_py) / n
        self.EPSILON = 1e-9

    def solve_forward_kinematics(self, l_known_vec: np.ndarray, 
            verbose: bool = False) -> tuple[float, float] | tuple[None, None]:
        X0 = self._get_initial_guess_fk(l_known_vec)
        solution, infodict, ier, mesg = fsolve(
            self._forward_kinematics_residuals, 
            X0, 
            args=(l_known_vec,),
            full_output=True,
            xtol=1.49012e-08
        )
        if ier == 1:
            return solution[0], solution[1]
        else:
            if verbose:
                print(f"--- FK solution failed ---")
                print(f"fsolve message: {mesg}")
            return None, None

    def solve_inverse_kinematics(self, kappa_b_target: float, gamma_g_target_rad: float, 
        verbose: bool = False) -> np.ndarray | None:
        X0 = self._get_initial_guess_ik(kappa_b_target, gamma_g_target_rad)

        bounds = self._get_ik_bounds()

        constraint_args = (kappa_b_target, gamma_g_target_rad)
        eq_constraints = {
            'type': 'eq',
            'fun': self._inverse_kinematics_constraints,
            'args': constraint_args
        }

        solution = minimize(
            self._objective_function,  
            X0,                        
            args=constraint_args,
            method='SLSQP',
            bounds=bounds,
            constraints=eq_constraints,
            options={'disp': verbose, 'maxiter': 2000} 
        )
        
        if solution.success:
            T_vec_sol = solution.x[0 : self.n]
            theta_0_vec_sol = solution.x[self.n : 2*self.n]
            kappa_c_vec_sol = solution.x[2*self.n : 3*self.n]
            
            kb = kappa_b_target 
            kappa_c_vec_safe = kappa_c_vec_sol + self.EPSILON
            
            l_vec_sol = (1.0 / kappa_c_vec_safe) * (self.L * kb - 2.0 * theta_0_vec_sol)

            if verbose and np.any(T_vec_sol < -1e-6):
                print("Warning: The solution contains negative T_i, which may not be physically realistic.")

            return l_vec_sol
        else:
            if not verbose:
                print(f"--- IK solution failed ---")
                print(f"Minimize message: {solution.message}")
            return None

    def _get_initial_guess_fk(self, l_known_vec: np.ndarray) -> np.ndarray:
        b = self.L - l_known_vec
        A = np.zeros((self.n, 2))
        A[:, 0] = self.L * self.d * np.cos(self.cable_angles)
        A[:, 1] = self.L * self.d * np.sin(self.cable_angles)
        try:
            xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            x, y = xy
            kappa_b_guess = np.sqrt(x**2 + y**2)
            gamma_g_guess = np.arctan2(y, x)
        except np.linalg.LinAlgError:
            kappa_b_guess = 0.0
            gamma_g_guess = 0.0
        beta_vec_guess = self.cable_angles - gamma_g_guess
        d_i_vec_guess = self.d * np.cos(beta_vec_guess)
        theta_0_vec_guess = np.zeros(self.n)
        kb_safe = kappa_b_guess + self.EPSILON
        kappa_c_vec_guess = kb_safe / (1 - kb_safe * d_i_vec_guess + self.EPSILON)
        T_vec_guess = np.full(self.n, 1.0) 
        X0 = np.concatenate([
            [kappa_b_guess, gamma_g_guess],
            T_vec_guess, theta_0_vec_guess, kappa_c_vec_guess
        ])
        return X0

    def _forward_kinematics_residuals(self, X: np.ndarray, l_known_vec: np.ndarray) -> np.ndarray:
        kappa_b = X[0]
        gamma_g = X[1]
        T_vec = X[2 : 2 + self.n]
        theta_0_vec = X[2 + self.n : 2 + 2*self.n]
        kappa_c_vec = X[2 + 2*self.n : 2 + 3*self.n]

        kb_safe = kappa_b + self.EPSILON
        kappa_c_vec_safe = kappa_c_vec + self.EPSILON

        phi_b = self.L * kappa_b
        alpha = phi_b / 2.0
        beta_vec = self.cable_angles - gamma_g
        d_i_vec = self.d * np.cos(beta_vec)

        residuals = np.zeros(3 * self.n + 2)

        residuals[0] = np.sum(T_vec * self.d * np.cos(theta_0_vec) * np.cos(beta_vec)) - self.Kb * kappa_b

        residuals[1] = np.sum(T_vec * self.d * np.cos(theta_0_vec) * np.sin(beta_vec))

        term_in_arcsin = (1 - kb_safe * d_i_vec) * (kappa_c_vec_safe / kb_safe) * np.sin(alpha)
        term_in_arcsin_clipped = np.clip(term_in_arcsin, -1.0, 1.0)
        residuals[2 : 2 + self.n] = theta_0_vec - (alpha - np.arcsin(term_in_arcsin_clipped))

        term1_vec = (1.0 / kb_safe - d_i_vec) * (1.0 - np.cos(alpha))
        term2_vec = (1.0 / kappa_c_vec_safe) * (1.0 - np.cos(alpha - theta_0_vec))
        residuals[2 + self.n : 2 + 2*self.n] = T_vec - self.Kc * (kb_safe / (kappa_c_vec_safe**2)) * (term1_vec - term2_vec)

        residuals[2 + 2*self.n : 2 + 3*self.n] = l_known_vec - (1.0 / kappa_c_vec_safe) * (self.L * kappa_b - 2.0 * theta_0_vec)
        return residuals

    def _get_initial_guess_ik(self, kappa_b_target: float, gamma_g_target_rad: float) -> np.ndarray:
        kb = kappa_b_target
        gg = gamma_g_target_rad
        
        beta_vec = self.cable_angles - gg
        d_i_vec = self.d * np.cos(beta_vec)
        
        T_vec_guess = np.zeros(self.n)
        j_slack = np.argmax(np.abs(beta_vec))
        active_indices = [i for i in range(self.n) if i != j_slack]
        
        if self.n > 2:
            A = np.zeros((2, self.n - 1))
            b = np.array([self.Kb * kb, 0.0])
            for idx, i in enumerate(active_indices):
                A[0, idx] = self.d * np.cos(beta_vec[i])
                A[1, idx] = self.d * np.sin(beta_vec[i])
            try:
                T_active, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                for idx, i in enumerate(active_indices):
                    T_vec_guess[i] = max(0.01, T_active[idx])
            except np.linalg.LinAlgError:
                T_vec_guess[active_indices] = 1.0
        else:
             T_vec_guess[active_indices] = 1.0
                
        theta_0_vec_guess = np.zeros(self.n)
        
        kb_safe = kb + self.EPSILON
        kappa_c_vec_guess = kb_safe / (1 - kb_safe * d_i_vec + self.EPSILON)

        X0 = np.concatenate([
            T_vec_guess, theta_0_vec_guess, kappa_c_vec_guess
        ])
        return X0

    def _objective_function(self, X: np.ndarray, *args) -> float:
        T_vec = X[0 : self.n]
        return np.sum(T_vec**2)


    def _get_ik_bounds(self) -> Bounds:
        T_bounds_list = [(0, None)] * self.n                     # T_i >= 0
        theta_0_bounds_list = [(-np.pi/2, np.pi/2)] * self.n
        kappa_c_bounds_list = [(self.EPSILON, None)] * self.n    # kappa_c,i > 0
        
        all_bounds_list = (
            T_bounds_list + 
            theta_0_bounds_list + 
            kappa_c_bounds_list
        )
        
        lower_bounds = [b[0] if b[0] is not None else -np.inf for b in all_bounds_list]
        upper_bounds = [b[1] if b[1] is not None else np.inf for b in all_bounds_list]
        
        return Bounds(lower_bounds, upper_bounds)

    def _inverse_kinematics_constraints(self, X: np.ndarray, kappa_b_target: float, gamma_g_target_rad: float) -> np.ndarray:
        kb = kappa_b_target
        gg = gamma_g_target_rad
        
        T_vec = X[0 : self.n]
        theta_0_vec = X[self.n : 2*self.n]
        kappa_c_vec = X[2*self.n : 3*self.n]
        
        kb_safe = kb + self.EPSILON
        kappa_c_vec_safe = kappa_c_vec + self.EPSILON
        
        phi_b = self.L * kb
        alpha = phi_b / 2.0
        
        beta_vec = self.cable_angles - gg
        d_i_vec = self.d * np.cos(beta_vec)

        residuals = np.zeros(2 * self.n + 2)
        
        residuals[0] = np.sum(T_vec * self.d * np.cos(theta_0_vec) * np.cos(beta_vec)) - self.Kb * kb
        
        residuals[1] = np.sum(T_vec * self.d * np.cos(theta_0_vec) * np.sin(beta_vec))

        term_in_arcsin = (1 - kb_safe * d_i_vec) * (kappa_c_vec_safe / kb_safe) * np.sin(alpha)
        term_in_arcsin_clipped = np.clip(term_in_arcsin, -1.0, 1.0)
        residuals[2 : 2 + self.n] = theta_0_vec - (alpha - np.arcsin(term_in_arcsin_clipped))
        
        term1_vec = (1.0 / kb_safe - d_i_vec) * (1.0 - np.cos(alpha))
        term2_vec = (1.0 / kappa_c_vec_safe) * (1.0 - np.cos(alpha - theta_0_vec))
        residuals[2 + self.n : 2 + 2*self.n] = T_vec - self.Kc * (kb_safe / (kb_safe**2)) * (term1_vec - term2_vec)
        
        return residuals

if __name__ == "__main__":
    arm_section = SoftArmSection(
        n=3,
        L=9.30,   # cm
        d=1.25,   # cm
        Kb=20.02, # N*cm^2
        Kc=3.10   # N/cm^2
    )

    kappa_target = 0.15
    gamma_target_deg = 45.0
    gamma_target_rad = np.radians(gamma_target_deg)

    print(f"--- Test: Inverse Kinematics (IK) ---")
    print(f"Target: kappa_b = {kappa_target:.4f}, gamma_g = {gamma_target_deg:.2f}°")

    start_time_ik = time.time()
    l_target_vec = arm_section.solve_inverse_kinematics(
        kappa_target,
        gamma_target_rad,
        verbose=False
    )
    end_time_ik = time.time()

    if l_target_vec is not None:
        print(f"IK solution successful (time: {end_time_ik - start_time_ik:.4f} s)")
        print(f"l_i = {np.round(l_target_vec, 6)}")
        contraction = arm_section.L - l_target_vec
        print(f"Δl = {np.round(contraction, 6)}")
    else:
        print("IK solution failed.")

    print("\n" + "=" * 40 + "\n")

    print(f"--- Test: Forward Kinematics (FK) ---")
    print(f"Target: l_i = {np.round(l_target_vec, 6)}")

    start_time_fk = time.time()
    kappa_solved, gamma_solved_rad = arm_section.solve_forward_kinematics(
        l_target_vec,
        verbose=False
    )
    end_time_fk = time.time()

    if kappa_solved is not None:
        print(f"FK solution successful (time: {end_time_fk - start_time_fk:.4f} s)")
        gamma_solved_deg = np.degrees(gamma_solved_rad)
        print(f"kappa_b = {kappa_solved:.4f}, gamma_g = {gamma_solved_deg:.2f}°")
    else:
        print("FK solution failed.")

    print("\n" + "=" * 40 + "\n")

    print(f"--- Verification ---")
    print(f"Target: kappa_b = {kappa_target:.4f}, gamma_g = {gamma_target_deg:.2f}°")
    print(f"Solution: kappa_b = {kappa_solved:.4f}, gamma_g = {gamma_solved_deg:.2f}°")
    print(f"Error: kappa_error = {np.abs(kappa_target - kappa_solved):.2e}, gamma_error = {np.abs(gamma_target_deg - gamma_solved_deg):.2e}")
