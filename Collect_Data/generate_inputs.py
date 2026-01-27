import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Lib.SoftArm_lib import SoftArmSection

data = np.load("./Data/trajectory.npz", allow_pickle=True)
t = data["t"]
phi_b_deg_trajectory = data["phi_b_deg_trajectory"]
gamma_g_deg_trajectory = data["gamma_g_deg_trajectory"]

arm_section = SoftArmSection(
        n=3,
        L=9.30,   # cm
        d=1.25,   # cm
        Kb=20.02, # N*cm^2
        Kc=3.10   # N/cm^2
    )

l_target_trajectory = []

for i in range(len(t)):
    phi_b_deg = phi_b_deg_trajectory[i]
    gamma_g_deg = gamma_g_deg_trajectory[i]
    
    kappa_b = np.radians(phi_b_deg) / arm_section.L
    gamma_g_rad = np.radians(gamma_g_deg)
    
    l_target = arm_section.solve_inverse_kinematics(
        kappa_b, 
        gamma_g_rad, 
        verbose=False
    )
    
    if l_target is not None:
        l_target_trajectory.append(l_target)
    else:
        print(f"warning: Inverse kinematics solution failed at {i} time points.！")
        if len(l_target_trajectory) > 0:
            l_target_trajectory.append(l_target_trajectory[-1])
        else:
            l_target_trajectory.append(np.array([arm_section.L] * arm_section.n))

l_target_trajectory = np.array(l_target_trajectory)

# motor parameters
diameter = 10  # diameter of the pulley in mm
steps_per_rev = 3200  # number of steps per revolution of the stepper motor
circumference = diameter * np.pi 

L = arm_section.L  # cm
L_mm = L * 10 

motor_steps_trajectory = []

for i in range(len(l_target_trajectory)):
    l_i = l_target_trajectory[i]  
    l_i_mm = l_i * 10  
    
    delta_l = L_mm - l_i_mm  
    
    motor_steps = (delta_l / circumference) * steps_per_rev
    motor_steps = np.round(motor_steps).astype(int)  # round to nearest integer
    # print(f"Time {t[i]:.2f}s: target length {l_i}, length change {delta_l}, motor steps {motor_steps}")
    motor_steps_trajectory.append(motor_steps)

motor_steps_trajectory = np.array(motor_steps_trajectory)

output_file = "./Data/control_inputs.npz"
np.savez(
    output_file,
    t=t,
    motor_steps_trajectory=motor_steps_trajectory,
)
print(f"\nResults saved to {output_file}")
