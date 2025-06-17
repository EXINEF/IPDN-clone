import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


RES_FILE_TO_ANALYZE = [
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_res_SOTA.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/new-logs/training_roberta_res_SOTA_1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5__1.err",


    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_2_res_3new_random.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_res_3new.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_asymmetricloss.err",

]

GRES_FILE_TO_ANALYZE = [
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_gres_SOTA.err",

        "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_asymmetricloss.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight_1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_asymmetricloss_1.err",

    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_high_asymmetricloss.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_extreme_asymmetricloss.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_clip_gres_posweight.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_7new_clip_norm.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_3_gres_11_base_attention_1.err",
        "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_7new_clip_norm.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_3_gres_11_base_attention_1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_attnpool.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_attnpool_1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_mean.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_mean_1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_posweight_roberta_objects.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_posweight_roberta_objects_1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight_1.err",
]

def parse_log_file(filepath):
    epoch_data = defaultdict(lambda: {'loss': [], 'scene_loss': []})
    
    with open(filepath, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'INFO' in line:
                epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    
                    loss_match = re.search(r'loss: ([\d.-]+)', line)
                    if loss_match:
                        epoch_data[epoch]['loss'].append(float(loss_match.group(1)))
                    
                    scene_loss_match = re.search(r'scene_loss: ([\d.-]+)', line)
                    if scene_loss_match:
                        epoch_data[epoch]['scene_loss'].append(float(scene_loss_match.group(1)))
    
    epochs = sorted(epoch_data.keys())
    avg_loss = [np.mean(epoch_data[e]['loss']) if epoch_data[e]['loss'] else np.nan for e in epochs]
    avg_scene_loss = [np.mean(epoch_data[e]['scene_loss']) if epoch_data[e]['scene_loss'] else np.nan for e in epochs]
    
    return epochs, avg_loss, avg_scene_loss

def plot_individual_file(filepath, output_dir):
    epochs, loss, scene_loss = parse_log_file(filepath)
    filename = os.path.basename(filepath).replace('.log', '')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'Loss Curves - {filename}')
    
    ax1.plot(epochs, loss, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss vs Epoch')
    ax1.grid(True, alpha=0.3)
    
    if not all(np.isnan(scene_loss)):
        ax2.plot(epochs, scene_loss, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Scene Loss')
        ax2.set_title('Scene Loss vs Epoch')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No scene_loss data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Scene Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=150)
    plt.close()

def plot_comparative(files, dataset_name, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Comparative Loss Curves - {dataset_name}')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    
    for i, filepath in enumerate(files):
        epochs, loss, scene_loss = parse_log_file(filepath)
        filename = os.path.basename(filepath).replace('.log', '')
        
        ax1.plot(epochs, loss, color=colors[i], linewidth=2, label=filename)
        if not all(np.isnan(scene_loss)):
            ax2.plot(epochs, scene_loss, color=colors[i], linewidth=2, label=filename)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Scene Loss')
    ax2.set_title('Scene Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_comparative.png'), dpi=150)
    plt.close()

def main():
    base_dir = 'tmp/plot_eval_losses'
    gres_dir = os.path.join(base_dir, 'GRES')
    res_dir = os.path.join(base_dir, 'RES')
    
    os.makedirs(gres_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    if GRES_FILE_TO_ANALYZE:
        for filepath in GRES_FILE_TO_ANALYZE:
            if os.path.exists(filepath):
                plot_individual_file(filepath, gres_dir)
        
        plot_comparative(GRES_FILE_TO_ANALYZE, 'GRES', gres_dir)
    
    if RES_FILE_TO_ANALYZE:
        for filepath in RES_FILE_TO_ANALYZE:
            if os.path.exists(filepath):
                plot_individual_file(filepath, res_dir)
        
        plot_comparative(RES_FILE_TO_ANALYZE, 'RES', res_dir)

if __name__ == '__main__':
    main()