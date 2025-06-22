import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import shutil


RES_FILE_TO_ANALYZE = [
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_res_SOTA.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/new-logs/training_roberta_res_SOTA_1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5__1.err",


    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_2_res_3new_random.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_res_3new.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_asymmetricloss.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_res_gt.err",

]

GRES_FILE_TO_ANALYZE = [
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_gres_SOTA.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1.err",

    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_512.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_weight5.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_focalloss_weight20.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_gres_gt.err",
]


def parse_log_file(filepath):
    epoch_data = defaultdict(lambda: {'loss': [], 'scene_loss': []})
    if not os.path.exists(filepath):
        print(f"WARNING: Log file not found: {filepath}")
        return [], [], []
    
    print(f"Parsing log file: {filepath}")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if 'Epoch' in line and 'INFO' in line:
                    epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        
                        # More robust regex that handles scientific notation and ensures we get the right "loss"
                        loss_match = re.search(r'(?:^|,\s)loss:\s*([\d.-]+(?:[eE][+-]?\d+)?)', line)
                        if loss_match:
                            epoch_data[epoch]['loss'].append(float(loss_match.group(1)))
                        
                        scene_loss_match = re.search(r'scene_loss:\s*([\d.-]+(?:[eE][+-]?\d+)?)', line)
                        if scene_loss_match:
                            epoch_data[epoch]['scene_loss'].append(float(scene_loss_match.group(1)))
        
        # Check if we found any data
        if not epoch_data:
            print(f"WARNING: No loss data found in file: {filepath}")
            return [], [], []
        
        epochs = sorted(epoch_data.keys())
        avg_loss = [np.mean(epoch_data[e]['loss']) if epoch_data[e]['loss'] else np.nan for e in epochs]
        avg_scene_loss = [np.mean(epoch_data[e]['scene_loss']) if epoch_data[e]['scene_loss'] else np.nan for e in epochs]
        
        print(f"  Found {len(epochs)} epochs with loss data")
        return epochs, avg_loss, avg_scene_loss
        
    except Exception as e:
        print(f"ERROR parsing {filepath}: {str(e)}")
        return [], [], []


def plot_individual_file(filepath, output_dir):
    epochs, loss, scene_loss = parse_log_file(filepath)
    
    if not epochs:  # Skip if no data
        print(f"  Skipping plot for {filepath} - no data found")
        return False
    
    # Fix the filename handling - properly replace .err extension
    filename = os.path.basename(filepath)
    if filename.endswith('.err'):
        filename = filename[:-4]  # Remove .err
    elif filename.endswith('.log'):
        filename = filename[:-4]  # Remove .log
    
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
    output_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {output_path}")
    return True


def plot_comparative(files, dataset_name, output_dir):
    # Filter out files that don't exist or have no data
    valid_files = []
    for filepath in files:
        if os.path.exists(filepath):
            epochs, _, _ = parse_log_file(filepath)
            if epochs:
                valid_files.append(filepath)
    
    if not valid_files:
        print(f"  No valid files with data for comparative plot of {dataset_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Comparative Loss Curves - {dataset_name}')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_files)))
    
    for i, filepath in enumerate(valid_files):
        epochs, loss, scene_loss = parse_log_file(filepath)
        filename = os.path.basename(filepath)
        if filename.endswith('.err'):
            filename = filename[:-4]
        elif filename.endswith('.log'):
            filename = filename[:-4]
        
        ax1.plot(epochs, loss, color=colors[i], linewidth=2, label=filename)
        if not all(np.isnan(scene_loss)):
            ax2.plot(epochs, scene_loss, color=colors[i], linewidth=2, label=filename)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Comparison')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Scene Loss')
    ax2.set_title('Scene Loss Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{dataset_name}_comparative.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved comparative plot: {output_path}")


def main():
    base_dir = 'tmp/plot_eval_losses'
    
    # Clean up old results using shutil for more reliable deletion
    if os.path.exists(base_dir):
        print(f"Removing old results directory: {base_dir}")
        shutil.rmtree(base_dir)
    
    # Create directories
    gres_dir = os.path.join(base_dir, 'GRES')
    res_dir = os.path.join(base_dir, 'RES')
    
    os.makedirs(gres_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    print(f"Created directories: {gres_dir}, {res_dir}")
    
    # Process GRES files
    if GRES_FILE_TO_ANALYZE:
        print("\nProcessing GRES files...")
        successful_plots = 0
        for filepath in GRES_FILE_TO_ANALYZE:
            if plot_individual_file(filepath, gres_dir):
                successful_plots += 1
        
        print(f"Created {successful_plots} individual GRES plots")
        
        if successful_plots > 0:
            plot_comparative(GRES_FILE_TO_ANALYZE, 'GRES', gres_dir)
    
    # Process RES files
    if RES_FILE_TO_ANALYZE:
        print("\nProcessing RES files...")
        successful_plots = 0
        for filepath in RES_FILE_TO_ANALYZE:
            if plot_individual_file(filepath, res_dir):
                successful_plots += 1
        
        print(f"Created {successful_plots} individual RES plots")
        
        if successful_plots > 0:
            plot_comparative(RES_FILE_TO_ANALYZE, 'RES', res_dir)
    
    # List created files
    print("\nCreated files:")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")


if __name__ == '__main__':
    main()