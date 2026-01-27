import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Lib.SoftArm_lib import SoftArmSection, constant_curvature
from Lib.DeePC_lib import hankel_matrix


data = np.load("./Data/control_inputs.npz", allow_pickle=True)
t = data["t"]
control_inputs = data["motor_steps_trajectory"]  # shape: (T, n)

arm_section = SoftArmSection(
        n=3,
        L=9.30,   # cm
        d=1.25,   # cm
        Kb=20.02, # N*cm^2
        Kc=3.10   # N/cm^2
    )

diameter = 10  # pulley diameter, unit: mm
steps_per_rev = 3200  # number of steps per revolution
circumference = diameter * np.pi  # circumference, unit: mm

mm_per_step = circumference / steps_per_rev

l_trajectory = arm_section.L - (control_inputs * mm_per_step / 10.0)

num_steps = l_trajectory.shape[0]
position_trajectory = np.zeros((num_steps, 3))

for i in range(num_steps):
    l_vec = l_trajectory[i, :]
    kappa_b, gamma_g = arm_section.solve_forward_kinematics(l_vec, verbose=False)
    
    if kappa_b is not None and gamma_g is not None:
        theta = kappa_b * arm_section.L
        x, y, z = constant_curvature(theta, gamma_g, arm_section.L * 10.0)
        # noise = np.random.uniform(-0.002, 0.002, 3)
        noise = np.random.normal(0, 0.002, 3)
        position_trajectory[i, :] = [x + noise[0], y + noise[1], -z + noise[2]]
    else:
        print(f"warning: Forward kinematics solution failed at {i} time points.！")
        if i > 0:
            position_trajectory[i, :] = position_trajectory[i-1, :]
        else:
            position_trajectory[i, :] = [0, 0, -arm_section.L * 10.0]

Tini = 20
N = 25
L_hankel = Tini + N

u_data = control_inputs.T
y_data = position_trajectory.T  # shape: (3, T)

T = u_data.shape[1]
if L_hankel > T:
    raise ValueError(f"Data length is not enough! Need at least {L_hankel} time steps, but only {T} time steps.")

U_hankel = hankel_matrix(u_data, L_hankel)
Y_hankel = hankel_matrix(y_data, L_hankel)

print(f"U_hankel.shape = {U_hankel.shape}")
print(f"Y_hankel.shape = {Y_hankel.shape}")

m = u_data.shape[0]
p = y_data.shape[0]

Up = U_hankel[:m * Tini, :]
Uf = U_hankel[m * Tini:, :]

Yp = Y_hankel[:p * Tini, :]
Yf = Y_hankel[p * Tini:, :]


AA = np.vstack([Up, Yp, Uf, Yf])
print(f"\nOriginal Hankel matrix size: AA.shape = {AA.shape}")

U_aa, S_aa, Vh_aa = np.linalg.svd(AA, full_matrices=True)
V_aa = Vh_aa.T

data_length = 600
data_length = min(data_length, V_aa.shape[1])

AA_L = AA @ V_aa[:, :data_length]

row_Up = Up.shape[0]
row_Yp = Yp.shape[0]
row_Uf = Uf.shape[0]

Up = AA_L[:row_Up, :]
Yp = AA_L[row_Up:row_Up+row_Yp, :]
Uf = AA_L[row_Up+row_Yp:row_Up+row_Yp+row_Uf, :]
Yf = AA_L[row_Up+row_Yp+row_Uf:, :]

print(f"\nUp.shape = {Up.shape}  (past input)")
print(f"Uf.shape = {Uf.shape}  (future input)")
print(f"Yp.shape = {Yp.shape}  (past output)")
print(f"Yf.shape = {Yf.shape}  (future output)")

output_file = "./Data/hankel_matrices.npz"
np.savez(
    output_file,
    t=t,
    control_inputs=control_inputs,
    position_trajectory=position_trajectory,
    U_hankel=U_hankel,
    Y_hankel=Y_hankel,
    Up=Up,
    Uf=Uf,
    Yp=Yp,
    Yf=Yf,
    Tini=Tini,
    N=N,
    m=m,
    p=p
)
print(f"Results saved to {output_file}")

print(f"Rank of U_hankel: {np.linalg.matrix_rank(U_hankel)}")
print(f"Rank of Y_hankel: {np.linalg.matrix_rank(Y_hankel)}")

combined_hankel = np.vstack([U_hankel, Y_hankel])
print(f"Rank of [U; Y] concatenated matrix: {np.linalg.matrix_rank(combined_hankel)}")
print(f"Expected full rank: {(m + p) * L_hankel}")
