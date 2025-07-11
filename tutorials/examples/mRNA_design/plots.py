import matplotlib.pyplot as plt
import torch


def plot_training_curves(
    loss_history, reward_components, out_path="training_curves.png"
):

    reward_tensor = torch.tensor(reward_components)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle("Training Curves", fontsize=18)

    # Plot Loss
    axes[0].plot(loss_history, linewidth=2)
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)

    # Plot Reward Components
    labels = ["GC", "MFE", "CAI"]
    colors = ["green", "blue", "orange"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[i + 1].plot(reward_tensor[:, i], label=label, color=color, linewidth=2)
        axes[i + 1].set_title(f"{label} Evolution", fontsize=14)
        axes[i + 1].set_xlabel("Iteration", fontsize=12)
        axes[i + 1].legend()

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.savefig(out_path, dpi=300)
    plt.close()
