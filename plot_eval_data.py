import re
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime


FILES_TO_ANALYZE = [
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_res_SOTA.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight_1.err",

    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_res_gt.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_res_gt_1.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_res_gt_pe.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_res_gt_pe_1.err",   
    
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_res_SOTA1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_res_SOTA_1.err",

    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_gres_SOTA.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1.err",

    #"/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_512.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_weight5.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_focalloss_weight20.err",

    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_gres_gt.err",
    # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_gres_gt_1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_gres_gt_pe.err",
    #     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/training_roberta_gres_gt_pe_1.err",

    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCE-GT-PE-0_1-LR.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCE-PW-GT-PE-0_1-LR.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCE-GT-PE-0_01-LR.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCE-PW-GT-PE-0_01-LR.err",
    

    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCE-PW-GT-PE-0_1-LR_1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCE-PW-GT-PE-0_01-LR_1.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-AL-GT-PE-0_01-LR-COMPLEX.err",
    # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new2/GRES-OA-BCEPW-GT-PE-0_01-LR-COMPLEX.err",

    # CLIP BASED
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new3/GRES-LongClip.err",
    "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_clip_gres_posweight.err",
    ]


# FILES_TO_ANALYZE = [
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/RES-baseline.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/RES-OA-BCE-PW.err",
    

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-baseline.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-OA-BCE.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-OA-BCE-PW.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-OA-BCE-PW-NoReLU.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-OA-BCE-PW-NoReLU-512.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-OA-BCE-PW-NoReLU-W5.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-thesis/GRES-OA-Focal-W20.err",
    
# ]




# FILES_TO_ANALYZE = [
   
#     ###
#     ### RES
#     ###
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_res_SOTA.err",   # BEST CHECKPOINT "/nfs/data_todi/jli/Alessio_works/IPDN-clone/exps/default_res/20250603_215236/best.pth"
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/new-logs/training_roberta_res_SOTA_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5__1.err",


#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_2_res_3new_random.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_res_3new.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_res_2new.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_asymmetricloss.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_asymmetricloss_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_res_SOTA.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_res_SOTA_1.err",
#     ###
#     ### GRES
#     ###

#     ## baseline
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_gres_SOTA.err", ### "/nfs/data_todi/jli/Alessio_works/IPDN-clone/exps/default_gres/20250530_140611/best.pth",

#     # TOP PERFOMERS
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_posweight_roberta_objects.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_mean.err",

#     # CLIP BASED
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_clip_gres_posweight.err",

#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_attnpool.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_attnpool_1.err",


#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_mean_1.err",
    
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_posweight_roberta_objects_1.err",

#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1_posweight.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_512.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_weight5.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_focalloss_weight20.err",

    
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_512_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_weight5_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_focalloss_weight20_1.err",

#     ]


# FILES_TO_ANALYZE = [
   
#     ###
#     ### RES
#     ###
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_res_SOTA.err",   # BEST CHECKPOINT "/nfs/data_todi/jli/Alessio_works/IPDN-clone/exps/default_res/20250603_215236/best.pth"
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/new-logs/training_roberta_res_SOTA_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_new_loss1_5__1.err",


#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_2_res_3new_random.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_res_3new.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_res_2new.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_asymmetricloss.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_posweight_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_res_2new_loss1_asymmetricloss_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_res_SOTA.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_res_SOTA_1.err",
#     ###
#     ### GRES
#     ###

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/training_roberta_gres_SOTA.err", ### "/nfs/data_todi/jli/Alessio_works/IPDN-clone/exps/default_gres/20250530_140611/best.pth",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_roberta_gres_SOTA_1.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1.err",
#     #"/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1_5.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1__1.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1_5__1.err",
#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_2_gres_3new.err",
#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_gres_3new.err",

#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss2_gres_3new.err",
#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_5_gres_3new.err",

#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_gres_2new.err",
    


#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss10_gres_7new.err",


#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_7new_roberta_norm.err",
#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_7new_clip_norm.err",
#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_7new_roberta.err",
#     # # # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_6new_roberta.err",
    
#     #"/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_3_gres_11_base.err",
#     #"/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_11_base.err",
#     #"/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_3_gres_11_base_attention.err",


#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight_1.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_asymmetricloss_1.err",

#     # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_posweight.err",
#     # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_asymmetricloss.err",
#     # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_high_asymmetricloss.err",
#     # # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_extreme_asymmetricloss.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_clip_gres_posweight.err",

#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_high_asymmetricloss_1.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_2new_loss1_extreme_asymmetricloss_1.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_gres_7new_clip_norm.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs/training_test_loss1_3_gres_11_base_attention_1.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_attnpool.err",
#     # "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_attnpool_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_mean.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_soweight_mean_1.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_posweight_roberta_objects.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_posweight_roberta_objects_1.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new/training_roberta_gres_new_loss1_posweight.err",

#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_512.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_weight5.err",
#     "/nfs/data_todi/jli/Alessio_works/IPDN-clone/logs-new1/training_roberta_gres_2new_posweight_norelu_focalloss_weight20.err",

#     ]


def get_dataset_type(file_path):
    """Determine if a file is for 'gres' or 'res' dataset based on filename."""
    file_name = os.path.basename(file_path).lower()
    if 'gres' in file_name:
        return 'GRES'
    else:
        return 'RES'

def extract_metrics_all_epochs(file_path):
    """
    Extract evaluation metrics from a log file for all epochs, handling different output formats.
    
    Returns a list of dictionaries, each containing metrics for a single epoch evaluation:
    - timestamp: When the evaluation was performed
    - epoch: The epoch number 
    - miou: The mean intersection over union value
    - precision_25: Precision at 0.25 IoU threshold (overall or Acc_25)
    - precision_50: Precision at 0.50 IoU threshold (overall or Acc_50)
    - dataset_type: Whether the file is for 'GRES' or 'RES' dataset
    """
    all_results = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Determine dataset type
        dataset_type = get_dataset_type(file_path)
        
        # Split content by evaluation sections
        # Look for patterns that indicate the start of an evaluation
        eval_sections = re.split(r'Evaluate referring segmentation', content)
        
        current_epoch = None
        
        # Process the entire content first to extract all epoch numbers
        epoch_matches = re.findall(r'Epoch \[(\d+)/\d+\]', content)
        epoch_numbers = [int(epoch) for epoch in epoch_matches]
        
        # Process each evaluation section (skip the first section which is before any evaluation)
        for i, section in enumerate(eval_sections[1:], 1):
            # Try to find timestamp for this evaluation
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', 
                                     eval_sections[i-1][-30:] + section[:50])
            
            # Create a result dictionary for this epoch
            result = {
                'timestamp': None,
                'epoch': None,
                'miou': None,
                'precision_25': None,
                'precision_50': None,
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'dataset_type': dataset_type
            }
            
            # Set timestamp if found
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                result['timestamp'] = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            
            # Find epoch number for this evaluation
            # Look for the closest epoch number before this evaluation
            section_start_pos = len(''.join(eval_sections[:i]))
            preceding_content = content[:section_start_pos]
            epoch_matches_before = re.findall(r'Epoch \[(\d+)/\d+\]', preceding_content)
            
            if epoch_matches_before:
                result['epoch'] = int(epoch_matches_before[-1])
                current_epoch = result['epoch']
            else:
                # If we can't find an epoch number, use the previous one
                result['epoch'] = current_epoch
            
            # Extract mIoU value
            miou_match = re.search(r'mIoU : ([\d\.]+)', section)
            if miou_match:
                result['miou'] = float(miou_match.group(1))
            
            if dataset_type == 'GRES':
                # Try to extract precision values from GRES format (table with "overall" column)
                precision_pattern = r'INFO - (0\.\d+)\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)'
                precision_matches = re.findall(precision_pattern, section)
                
                if precision_matches:
                    for threshold, value in precision_matches:
                        if threshold == '0.25':
                            result['precision_25'] = float(value)
                        elif threshold == '0.50':
                            result['precision_50'] = float(value)
            else:  # RES format
                # Extract Acc_50 and Acc_25 from RES format
                acc_match = re.search(r'Acc_50: ([\d\.]+)\. Acc_25: ([\d\.]+)', section)
                if acc_match:
                    result['precision_50'] = float(acc_match.group(1)) * 100  # Convert to percentages to match GRES format
                    result['precision_25'] = float(acc_match.group(2)) * 100  # Convert to percentages to match GRES format
            
            # Add this epoch's results if we have any metrics
            if any(result.get(m) is not None for m in ['miou', 'precision_25', 'precision_50']):
                all_results.append(result)
        
        return all_results
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def plot_metrics(all_results):
    """Plot metrics from all analyzed files across all epochs, separating by dataset type."""
    output_dir = 'tmp/plot_eval_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort results by epoch if available, otherwise by timestamp
    sort_key = 'epoch' if all(r['epoch'] is not None for r in all_results) else 'timestamp'
    sorted_results = sorted(all_results, key=lambda x: (x['file_path'], x[sort_key] if x[sort_key] is not None else float('inf')))
    
    # Group results by dataset type and file
    dataset_grouped_results = {'RES': {}, 'GRES': {}}
    
    for result in sorted_results:
        dataset_type = result['dataset_type']
        file_path = result['file_path']
        if file_path not in dataset_grouped_results[dataset_type]:
            dataset_grouped_results[dataset_type][file_path] = []
        dataset_grouped_results[dataset_type][file_path].append(result)
    
    # Plot metrics for each dataset type
    for dataset_type, file_grouped_results in dataset_grouped_results.items():
        if not file_grouped_results:
            continue
            
        # Plot mIoU for each dataset type
        plt.figure(figsize=(14, 8))
        
        for file_path, results in file_grouped_results.items():
            file_name = os.path.basename(file_path)
            x_values = [r['epoch'] for r in results if r['epoch'] is not None and r['miou'] is not None]
            y_values = [r['miou'] for r in results if r['epoch'] is not None and r['miou'] is not None]
            
            if y_values:  # Only plot if we have valid data
                plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, label=file_name)
        
        plt.title(f'{dataset_type} mIoU across Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mIoU', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset_type}_miou_plot_all_epochs.png', dpi=300)
        
        # Plot Precision@0.25 for each dataset type
        plt.figure(figsize=(14, 8))
        
        for file_path, results in file_grouped_results.items():
            file_name = os.path.basename(file_path)
            x_values = [r['epoch'] for r in results if r['epoch'] is not None and r['precision_25'] is not None]
            y_values = [r['precision_25'] for r in results if r['epoch'] is not None and r['precision_25'] is not None]
            
            if y_values:  # Only plot if we have valid data
                plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, label=file_name)
        
        plt.title(f'{dataset_type} Precision@0.25 across Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Precision@0.25 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset_type}_precision_25_plot_all_epochs.png', dpi=300)
        
        # Plot Precision@0.50 for each dataset type
        plt.figure(figsize=(14, 8))
        
        for file_path, results in file_grouped_results.items():
            file_name = os.path.basename(file_path)
            x_values = [r['epoch'] for r in results if r['epoch'] is not None and r['precision_50'] is not None]
            y_values = [r['precision_50'] for r in results if r['epoch'] is not None and r['precision_50'] is not None]
            
            if y_values:  # Only plot if we have valid data
                plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, label=file_name)
        
        plt.title(f'{dataset_type} Precision@0.50 across Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Precision@0.50 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{dataset_type}_precision_50_plot_all_epochs.png', dpi=300)
    
    plt.close('all')  # Close all figures to avoid display issues

def main():
    all_results = []
    
    if not FILES_TO_ANALYZE:
        print("No files specified in FILES_TO_ANALYZE. Please add file paths to analyze.")
        return
    
    # Process each file
    for file_path in FILES_TO_ANALYZE:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            file_results = extract_metrics_all_epochs(file_path)
            if file_results:
                all_results.extend(file_results)
                print(f"Found {len(file_results)} evaluation results in {file_path}")
            else:
                print(f"No valid metrics found in {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    if all_results:
        # Filter out entries with no metrics at all
        all_results = [r for r in all_results if any(r.get(m) is not None 
                                               for m in ['miou', 'precision_25', 'precision_50'])]
        
        if all_results:
            # Plot metrics separately for each dataset type
            plot_metrics(all_results)
            
            # Group results by dataset type
            res_results = [r for r in all_results if r['dataset_type'] == 'RES']
            gres_results = [r for r in all_results if r['dataset_type'] == 'GRES']
            
            # Print metrics summary for RES dataset
            if res_results:
                print("\nRES Dataset Metrics Summary:")
                print("-" * 100)
                print(f"{'File Name':<30} {'Epoch':<10} {'mIoU':<10} {'Prec@0.25':<15} {'Prec@0.50':<15}")
                print("-" * 100)
                
                # Group by file and sort by epoch
                for file_path in sorted(set([r['file_path'] for r in res_results])):
                    file_results = [r for r in res_results if r['file_path'] == file_path]
                    file_results.sort(key=lambda x: x['epoch'] if x['epoch'] is not None else float('inf'))
                    
                    for result in file_results:
                        epoch = str(result['epoch']) if result['epoch'] is not None else 'N/A'
                        miou = f"{result['miou']:.4f}" if result['miou'] is not None else 'N/A'
                        prec25 = f"{result['precision_25']:.1f}" if result['precision_25'] is not None else 'N/A'
                        prec50 = f"{result['precision_50']:.1f}" if result['precision_50'] is not None else 'N/A'
                        
                        print(f"{result['file_name']:<30} {epoch:<10} {miou:<10} {prec25:<15} {prec50:<15}")
                    
                    # Add separation between files
                    print("-" * 100)
            
            # Print metrics summary for GRES dataset
            if gres_results:
                print("\nGRES Dataset Metrics Summary:")
                print("-" * 100)
                print(f"{'File Name':<30} {'Epoch':<10} {'mIoU':<10} {'Prec@0.25':<15} {'Prec@0.50':<15}")
                print("-" * 100)
                
                # Group by file and sort by epoch
                for file_path in sorted(set([r['file_path'] for r in gres_results])):
                    file_results = [r for r in gres_results if r['file_path'] == file_path]
                    file_results.sort(key=lambda x: x['epoch'] if x['epoch'] is not None else float('inf'))
                    
                    for result in file_results:
                        epoch = str(result['epoch']) if result['epoch'] is not None else 'N/A'
                        miou = f"{result['miou']:.4f}" if result['miou'] is not None else 'N/A'
                        prec25 = f"{result['precision_25']:.1f}" if result['precision_25'] is not None else 'N/A'
                        prec50 = f"{result['precision_50']:.1f}" if result['precision_50'] is not None else 'N/A'
                        
                        print(f"{result['file_name']:<30} {epoch:<10} {miou:<10} {prec25:<15} {prec50:<15}")
                    
                    # Add separation between files
                    print("-" * 100)
            
            # Print the best mIoU summaries for both datasets at the end
            print("\n======================= BEST RESULTS SUMMARY =======================")
            
            # Print best mIoU for RES dataset
            if res_results:
                print("\nRES Dataset Best mIoU Results:")
                print("-" * 100)
                print(f"{'File Name':<30} {'Best mIoU':<15} {'Epoch':<10} {'Prec@0.25':<15} {'Prec@0.50':<15}")
                print("-" * 100)
                
                for file_path in sorted(set([r['file_path'] for r in res_results])):
                    file_results = [r for r in res_results if r['file_path'] == file_path and r['miou'] is not None]
                    if file_results:
                        # Find the result with the best mIoU
                        best_result = max(file_results, key=lambda x: x['miou'])
                        
                        epoch = str(best_result['epoch']) if best_result['epoch'] is not None else 'N/A'
                        miou = f"{best_result['miou']:.4f}" if best_result['miou'] is not None else 'N/A'
                        prec25 = f"{best_result['precision_25']:.1f}" if best_result['precision_25'] is not None else 'N/A'
                        prec50 = f"{best_result['precision_50']:.1f}" if best_result['precision_50'] is not None else 'N/A'
                        
                        print(f"{best_result['file_name']:<30} {miou:<15} {epoch:<10} {prec25:<15} {prec50:<15}")
                
                print("-" * 100)
            
            # Print best mIoU for GRES dataset
            if gres_results:
                print("\nGRES Dataset Best mIoU Results:")
                print("-" * 100)
                print(f"{'File Name':<30} {'Best mIoU':<15} {'Epoch':<10} {'Prec@0.25':<15} {'Prec@0.50':<15}")
                print("-" * 100)
                
                for file_path in sorted(set([r['file_path'] for r in gres_results])):
                    file_results = [r for r in gres_results if r['file_path'] == file_path and r['miou'] is not None]
                    if file_results:
                        # Find the result with the best mIoU
                        best_result = max(file_results, key=lambda x: x['miou'])
                        
                        epoch = str(best_result['epoch']) if best_result['epoch'] is not None else 'N/A'
                        miou = f"{best_result['miou']:.4f}" if best_result['miou'] is not None else 'N/A'
                        prec25 = f"{best_result['precision_25']:.1f}" if best_result['precision_25'] is not None else 'N/A'
                        prec50 = f"{best_result['precision_50']:.1f}" if best_result['precision_50'] is not None else 'N/A'
                        
                        print(f"{best_result['file_name']:<30} {miou:<15} {epoch:<10} {prec25:<15} {prec50:<15}")
                
                print("-" * 100)
            
            print("\nAnalysis complete. Plots saved to tmp/plot_eval_data/")
        else:
            print("No valid metrics found in any file.")
    else:
        print("No valid results found in any file.")

if __name__ == "__main__":
    main()