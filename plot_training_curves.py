import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_training_data(file_path):
    data = np.load(file_path)
    return {
        'return_list': data['return_list'],
        'mv_return': data['mv_return'],
        'action_1_ratio_list': data['action_1_ratio_list'],
        'mv_action_1_ratio': data['mv_action_1_ratio'],
        'episodes_list': data['episodes_list'],
        'rho': float(data['rho']),
        'num_episodes': int(data['num_episodes']),
        'best_test_reward': float(data['best_test_reward']),
        'seed_number': int(data['seed_number'])
    }

def plot_return_curves(save_dir, rho_list, algo_list, shade="minmax", save_path=None):
    algo_color_map = {
        "dqn": ("#c0392b", "#f5b7b1"),   # red - DQN
        "ppo": ("#2980b9", "#aed6f1"),   # blue - PPO
        "sac": ("#27ae60", "#a9dfbf"),   # green - SAC
        "td3": ("#f39c12", "#fdebd0"),   # orange - TD3
        "ddpg": ("#8e44ad", "#d7bde2"),  # purple - DDPG
        "a2c": ("#16a085", "#a3e4d7"),   # cyan - A2C
    }
    
    rho_line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    for algo in algo_list:
        for rho_idx, rho in enumerate(rho_list):
            pattern = f"{algo}_training_data_rho_{rho}_*.npz"
            
            paths = sorted(glob.glob(os.path.join(save_dir, pattern)))
            
            if len(paths) == 0:
                print(f"warning: algorithm={algo}, rho={rho} not found, skipping")
                continue
            
            curves = []
            for p in paths:
                data = np.load(p, allow_pickle=True)
                return_list = np.asarray(data["return_list"], dtype=float).squeeze()
                curves.append(return_list)
            
            min_len = min(len(c) for c in curves)
            curves = np.stack([c[:min_len] for c in curves], axis=0)
            
            x = np.arange(min_len)
            mean_curve = curves.mean(axis=0)
            
            if shade == "minmax":
                low = curves.min(axis=0)
                high = curves.max(axis=0)
            elif shade == "std":
                std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_curve)
                low = mean_curve - std
                high = mean_curve + std
            
            line_color, shade_color = algo_color_map[algo.lower()]
            line_style = rho_line_styles[rho_idx % len(rho_line_styles)]
            label = rf"{algo.upper()} ($\rho={rho}$)"
            
            plt.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
            
            plt.plot(x, mean_curve, color=line_color, linewidth=2.0, 
                     linestyle=line_style, label=label)
    
    plt.ylim(-350, -180)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.title("Training Returns Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"returns_comparison_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"returns_comparison_rho_{rho_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_mv_return_curves(save_dir, rho_list, algo_list, shade="minmax", save_path=None):
    algo_color_map = {
        "dqn": ("#c0392b", "#f5b7b1"),   # red - DQN
        "ppo": ("#2980b9", "#aed6f1"),   # blue - PPO
        "sac": ("#27ae60", "#a9dfbf"),   # green - SAC
        "td3": ("#f39c12", "#fdebd0"),   # orange - TD3
        "ddpg": ("#8e44ad", "#d7bde2"),  # purple - DDPG
        "a2c": ("#16a085", "#a3e4d7"),   # cyan - A2C
    }
    
    rho_line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    for algo in algo_list:
        for rho_idx, rho in enumerate(rho_list):
            pattern = f"{algo}_training_data_rho_{rho}_*.npz"
            
            paths = sorted(glob.glob(os.path.join(save_dir, pattern)))
            
            if len(paths) == 0:
                print(f"warning: algorithm={algo}, rho={rho} not found, skipping")
                continue
            
            curves = []
            for p in paths:
                data = np.load(p, allow_pickle=True)
                mv_return_list = np.asarray(data["mv_return"], dtype=float).squeeze()
                curves.append(mv_return_list)
            
            min_len = min(len(c) for c in curves)
            curves = np.stack([c[:min_len] for c in curves], axis=0)
            
            x = np.arange(min_len)
            mean_curve = curves.mean(axis=0)
            
            if shade == "minmax":
                low = curves.min(axis=0)
                high = curves.max(axis=0)
            elif shade == "std":
                std = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean_curve)
                low = mean_curve - std
                high = mean_curve + std
            
            line_color, shade_color = algo_color_map[algo.lower()]
            line_style = rho_line_styles[rho_idx % len(rho_line_styles)]
            label = rf"{algo.upper()} ($\rho={rho}$)"
            
            plt.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
            
            plt.plot(x, mean_curve, color=line_color, linewidth=2.0, 
                     linestyle=line_style, label=label)
    
    plt.ylim(-350, -180)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.title("Training Moving Average Returns Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"mv_returns_comparison_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"mv_returns_comparison_rho_{rho_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    save_dir = "./Saved_Training_Data"
    plot_return_curves(save_dir, rho_list=[0.1], algo_list=["dqn", "ppo"], shade="minmax")
    plot_mv_return_curves(save_dir, rho_list=[0.1], algo_list=["dqn", "ppo"], shade="minmax")