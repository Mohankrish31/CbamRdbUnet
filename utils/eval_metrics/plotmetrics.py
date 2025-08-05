import matplotlib.pyplot as plt
import numpy as np
def plot_average_metrics(cpsnr_list, ssim_list, lpips_list, ebcm_list, save_path="avg_metrics_bar.png"):
    metrics = ['C-PSNR', 'SSIM', 'LPIPS', 'EBCM']
    averages = [
        np.mean(cpsnr_list),
        np.mean(ssim_list),
        np.mean(lpips_list),
        np.mean(ebcm_list)
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, averages, color=['blue', 'green', 'red', 'purple'])
    plt.title('Average Evaluation Metrics')
    plt.ylabel('Value')
    plt.ylim(0, max(averages) * 1.2)
    # Annotate bars
    for bar, value in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
