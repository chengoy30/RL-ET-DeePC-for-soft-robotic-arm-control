import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
from Lib.rl_utils import moving_average

# IEEE single-column width: 3.5 in
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.0,
})

ALGO_COLORS = {
    "dqn": [
        ("#e74c3c", "#fadbd8"),   # light red   (small rho)
        ("#c0392b", "#f5b7b1"),   # medium red
        ("#922b21", "#e6b0aa"),   # dark red    (large rho)
    ],
    "ppo": [
        ("#5dade2", "#d6eaf8"),   # light blue  (small rho)
        ("#2980b9", "#aed6f1"),   # medium blue
        ("#1a5276", "#85c1e9"),   # dark blue   (large rho)
    ],
    "a2c": [
        ("#58d68d", "#d5f5e3"),   # light green (small rho)
        ("#27ae60", "#a9dfbf"),   # medium green
        ("#1e8449", "#82e0aa"),   # dark green  (large rho)
    ],
}

def load_training_data(file_path):
    data = np.load(file_path)
    return {
        'return_list': data['return_list'],
        'action_1_ratio_list': data['action_1_ratio_list'],
        'episodes_list': data['episodes_list'],
        'rho': float(data['rho']),
        'num_episodes': int(data['num_episodes']),
        'best_test_reward': float(data['best_test_reward']),
        'seed_number': int(data['seed_number'])
    }

def plot_return_curves(save_dir, rho_list, algo_list, shade="minmax", save_path=None):
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
                return_list = np.clip(return_list, -400, None)
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

            line_color, shade_color = ALGO_COLORS[algo][rho_idx % len(ALGO_COLORS[algo])]
            label = rf"{algo.upper()} ($\rho={rho}$)"

            plt.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
            plt.plot(x, mean_curve, color=line_color, linewidth=2.0, linestyle='-', label=label)

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(rho_list), fontsize=9, columnspacing=1.0)
    plt.tight_layout()
    
    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"returns_comparison_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"returns_comparison_rho_{rho_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_mv_return_curves(save_dir, rho_list, algo_list, shade="minmax", window_size=9, save_path=None):
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
                return_list = np.clip(return_list, -400, None)
                mv_return_list = moving_average(return_list, window_size)
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

            line_color, shade_color = ALGO_COLORS[algo][rho_idx % len(ALGO_COLORS[algo])]
            label = rf"{algo.upper()} ($\rho={rho}$)"

            plt.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
            plt.plot(x, mean_curve, color=line_color, linewidth=2.0, linestyle='-', label=label)

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Returns", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(rho_list), fontsize=9, columnspacing=1.0)
    plt.tight_layout()
    
    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"mv_returns_comparison_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"mv_returns_comparison_rho_{rho_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_action_1_ratio_curves(save_dir, rho_list, algo_list, shade="minmax", save_path=None):
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
                action_1_ratio_list = np.asarray(data["action_1_ratio_list"], dtype=float).squeeze()
                curves.append(action_1_ratio_list)

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

            line_color, shade_color = ALGO_COLORS[algo][rho_idx % len(ALGO_COLORS[algo])]
            label = rf"{algo.upper()} ($\rho={rho}$)"

            plt.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
            plt.plot(x, mean_curve, color=line_color, linewidth=2.0, linestyle='-', label=label)

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Action 1 Ratio", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(rho_list), fontsize=9, columnspacing=1.0)
    plt.tight_layout()
    
    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"action_1_ratio_comparison_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"action_1_ratio_comparison_rho_{rho_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_mv_action_1_ratio_curves(save_dir, rho_list, algo_list, shade="minmax", window_size=9, save_path=None):
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
                action_1_ratio_list = np.asarray(data["action_1_ratio_list"], dtype=float).squeeze()
                mv_action_1_ratio = moving_average(action_1_ratio_list, window_size)
                curves.append(mv_action_1_ratio)

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

            line_color, shade_color = ALGO_COLORS[algo][rho_idx % len(ALGO_COLORS[algo])]
            label = rf"{algo.upper()} ($\rho={rho}$)"

            plt.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
            plt.plot(x, mean_curve, color=line_color, linewidth=2.0, linestyle='-', label=label)

    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Action 1 Ratio", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(rho_list), fontsize=9, columnspacing=1.0)
    plt.tight_layout()
    
    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"mv_action_1_ratio_comparison_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"mv_action_1_ratio_comparison_rho_{rho_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_return_and_trigger_ratio_curves(save_dir, rho_list, algo_list, shade="minmax", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.8), dpi=300)

    for algo in algo_list:
        for rho_idx, rho in enumerate(rho_list):
            pattern = f"{algo}_training_data_rho_{rho}_*.npz"
            paths = sorted(glob.glob(os.path.join(save_dir, pattern)))

            if len(paths) == 0:
                print(f"warning: algorithm={algo}, rho={rho} not found, skipping")
                continue

            ret_curves, ratio_curves = [], []
            for p in paths:
                data = np.load(p, allow_pickle=True)
                ret = np.asarray(data["return_list"], dtype=float).squeeze()
                ret = np.clip(ret, -400, None)
                ret_curves.append(ret)
                ratio = np.asarray(data["action_1_ratio_list"], dtype=float).squeeze()
                ratio_curves.append(ratio)

            line_color, shade_color = ALGO_COLORS[algo][rho_idx % len(ALGO_COLORS[algo])]
            label = rf"{algo.upper()} ($\rho={rho}$)"

            for ax, curves_list, ylabel in [
                (ax1, ret_curves, "Returns"),
                (ax2, ratio_curves, "Trigger Ratio"),
            ]:
                min_len = min(len(c) for c in curves_list)
                curves_arr = np.stack([c[:min_len] for c in curves_list], axis=0)
                x = np.arange(min_len)
                mean_curve = curves_arr.mean(axis=0)

                if shade == "minmax":
                    low = curves_arr.min(axis=0)
                    high = curves_arr.max(axis=0)
                elif shade == "std":
                    std = curves_arr.std(axis=0, ddof=1) if curves_arr.shape[0] > 1 else np.zeros_like(mean_curve)
                    low = mean_curve - std
                    high = mean_curve + std

                ax.fill_between(x, low, high, color=shade_color, alpha=0.4, linewidth=0)
                ax.plot(x, mean_curve, color=line_color, linestyle='-', linewidth=0.5, label=label)
                ax.set_xlabel("Episodes")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

    ax1.set_ylim(-400, -100)
    ax2.set_ylim(0, 1)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.38)
    fig.legend(handles, labels, frameon=True, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=len(rho_list), columnspacing=0.6)

    if save_path is None:
        rho_str = "_".join([str(r) for r in rho_list])
        algo_str = "_".join([a for a in algo_list if a is not None]) if algo_list[0] is not None else ""
        if algo_str:
            save_path = os.path.join("./Figure", f"combined_{algo_str}_rho_{rho_str}.png")
        else:
            save_path = os.path.join("./Figure", f"combined_rho_{rho_str}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    save_dir = "./Saved_Training_Data"
    plot_return_and_trigger_ratio_curves(save_dir, rho_list=[0.1, 0.5, 1.0], algo_list=["dqn", "ppo", "a2c"], shade="minmax")
    # plot_return_curves(save_dir, rho_list=[0.1, 0.5, 1.0], algo_list=["dqn", "ppo", "a2c"], shade="minmax")
    # plot_mv_return_curves(save_dir, rho_list=[0.1, 0.5, 1.0], algo_list=["dqn", "ppo", "a2c"], shade="minmax", window_size=25)
    # plot_action_1_ratio_curves(save_dir, rho_list=[0.1, 0.5, 1.0], algo_list=["dqn", "ppo", "a2c"], shade="minmax")
    # plot_mv_action_1_ratio_curves(save_dir, rho_list=[0.1, 0.5, 1.0], algo_list=["dqn", "ppo", "a2c"], shade="minmax", window_size=25)