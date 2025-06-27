import functools
import json
import os
from transformers import CLIPTokenizer, CLIPTextModel
import random
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools
import numpy as np

from ipdn.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou

from .dec_new import DEC
from transformers import RobertaModel
from pointnet2.pointnet2_utils import FurthestPointSampling

LOSS_SCENE_OBJ_WEIGHT = 1

ASL_GAMMA_NEG = 4
ASL_GAMMA_POS = 0
ASL_CLIP = 0.05

TEMPLATES = [
    # Original templates
    "There is a {} in the scene",
    "A scene with {}",
    "A scene containing {}",
    "A room with {}",
    "A room containing {}",
    "There is {} in the scene",
    "There is {} in the room",
    "{} is present in the scene",
    "{} is present in the room",
    "In the scene there is a {}",
    "In the room there is a {}",
    "The scene contains {}",
    "The room contains {}",
    # Additional templates for better ensemble
    "A photo of a {} in a room",
    "A 3D scan of a room with {}",
    "Indoor scene with {}",
    "This room has a {}",
    "You can see a {} here",
    "A {} can be found in this room",
    "Look, there's a {} in the scene",
]

FEATURE_DIM = 32          # input superpoint dim
HIDDEN_DIM = 512          # internal projected dim
CLIP_DIM = 768            # frozen CLIP embedding dim


class ComplexSceneEncoder(nn.Module):
    def __init__(self, superpoints_features_dim, hidden_dim, clip_dim):
        super().__init__()

        self.attention_aggregator = AttentionAggregation(
                feature_dim=clip_dim, 
                num_heads=8
            )  

        self.superpoint_proj = DeepResidualMLP(
            input_dim=superpoints_features_dim,
            hidden_dim=hidden_dim,
            output_dim=768,
            num_blocks=4,
        )

        self.clip_adapter = CLIPAdapter(
            clip_dim=clip_dim,
            adapter_dim=64,
            initial_scale=0.1,
        )

        
    def forward(self, sp_feats_batch, gt_objects_names):
        """
        sp_feats_batch: list of [num_sp_i, feature_dim] tensors, len = B
        batch_object_names: list of list of object names present in each scene
        clip_text_embeddings: Dict[str, Tensor(768,)], frozen CLIP embeddings

        Input: Superpoints (1000×32) 
        ↓
        Deep Residual MLP (32→512, 20 blocks) 
        ↓
        Multi-Head Self-Attention Aggregation 
        ↓
        CLIP Adapter Layer 
        ↓
        Multi-Head Binary Classifiers 
        ↓
        Asymmetric Loss with Progressive Training
        """
        device = sp_feats_batch[0].device
        
        aggregated_features = []
        for sp_feats in sp_feats_batch:
            # Project superpoints
            sp_proj = self.superpoint_proj(sp_feats)
            # Use attention aggregation instead of mean
            scene_feat = self.attention_aggregator(sp_proj.unsqueeze(0)).squeeze(0)
            aggregated_features.append(scene_feat)
        
        scene_features = torch.stack(aggregated_features)
        
        clip_features = self.clip_adapter(self.object_prompt_features)

        scene_features = F.normalize(scene_features, dim=-1)
        logits = torch.matmul(scene_features, clip_features.T)

        targets = self.get_targets(gt_objects_names, device)

        return logits, targets


    def get_targets(self, gt_objects_names, device):
        """
        Create binary targets for each scene based on the object names
        """
        batch_size = len(gt_objects_names)
        targets = torch.zeros((batch_size, len(self.object_index)), device=device)
        for i, object_names in enumerate(gt_objects_names):
            for obj in object_names:
                if obj in self.object_index:
                    targets[i, self.object_index[obj]] = 1.0
                else:
                    raise ValueError(f"Object '{obj}' not found in object_index. Please check the object names in the dataset.")
        return targets   


class AttentionAggregation(nn.Module):
    def __init__(self, feature_dim=768, num_heads=8):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(feature_dim)
        print(f"Using Multihead Attention with feature_dim={feature_dim}, num_heads={num_heads}")
        
    def forward(self, superpoint_features):
        # superpoint_features: [batch, num_superpoints, feature_dim]
        attended, _ = self.multihead_attention(
            superpoint_features, superpoint_features, superpoint_features
        )
        attended = self.layer_norm(attended + superpoint_features)
        # Learnable weighted pooling
        weights = F.softmax(attended.mean(dim=-1), dim=-1)
        scene_repr = (attended * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(scene_repr, dim=-1)
    
class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=512, output_dim=768, num_blocks=12):
        super().__init__()
        self.initial_proj = nn.Linear(input_dim, hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        print(f"Using Deep Residual MLP with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, num_blocks={num_blocks}")
        
    def forward(self, x):
        x = self.initial_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.layer_norm(x)
        return self.final_proj(x)
    
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),  # GELU often works better than ReLU for deep networks
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
        print(f"Using Residual MLP Block with dim={dim}")
        
    def forward(self, x):
        return self.layer_norm(x + self.mlp(x))
    
class CLIPAdapter(nn.Module):
    def __init__(self, clip_dim=768, adapter_dim=64, initial_scale=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(clip_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, clip_dim)
        )
        self.scale = nn.Parameter(torch.ones(1) * initial_scale)
        print(f"Using CLIP Adapter with clip_dim={clip_dim}, adapter_dim={adapter_dim}, initial_scale={initial_scale}")
        
    def forward(self, clip_features):
        adapted = self.adapter(clip_features)
        return clip_features + adapted * self.scale



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        
    def forward(self, x, y):
        # Asymmetric focusing for positive and negative samples
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        
        # Asymmetric focusing
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        
        los_pos = los_pos * ((1 - pt0) ** self.gamma_pos)
        los_neg = los_neg * (pt1 ** self.gamma_neg)
        
        asymmetric_loss = -los_pos.sum() - los_neg.sum()
        return asymmetric_loss
   

class IPDN(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        sampling_module=None,
        dec=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        infer_mode='pos',
        fps_num=512,
        fix_module=[],
        task_type='res',
    ):
        super().__init__()
        self.task_type = task_type

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.decoder_param = dec
        self.fps_num = fps_num
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')

        self.sampling_module = sampling_module
        self.dec = DEC(**dec, sampling_module=sampling_module, in_channel=media)

        self.scene_encoder = ComplexSceneEncoder(
            superpoints_features_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            clip_dim=CLIP_DIM
        )

        # criterion
        self.criterion = Criterion(**criterion)
        self.scene_encoder_criterion = AsymmetricLoss(
            gamma_neg=ASL_GAMMA_NEG,
            gamma_pos= ASL_GAMMA_POS,
            clip= ASL_CLIP
        )
        print(f"Using Asymmetric Loss with gamma_neg={ASL_GAMMA_NEG}, gamma_pos={ASL_GAMMA_POS}, clip={ASL_CLIP}")

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.infer_mode = infer_mode
        self.init_weights()
        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(IPDN, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, gt_objects_names=None):
   
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)
            
        sp_feats_batch = []
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            sp_feats_batch.append(sp_feats[s:e])
        
        logits, targets = self.scene_encoder(sp_feats_batch, gt_objects_names)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='tr')
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d)
        
        loss, loss_dict = self.criterion(out, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels, dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets)
        
        scene_loss = self.scene_encoder_criterion(logits, targets)
        loss_dict['scene_loss'] = scene_loss * LOSS_SCENE_OBJ_WEIGHT
        loss += scene_loss * LOSS_SCENE_OBJ_WEIGHT
        # remove the loss called `loss` in the loss_dict
        loss_dict['loss'] = loss
        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, semantic_labels=None, instance_labels=None,  gt_objects_names=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats_batch = []
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            sp_feats_batch.append(sp_feats[s:e])

        logits, targets = self.scene_encoder(sp_feats_batch, gt_objects_names)
        
        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='pred')
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        
        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d) 
        ret = self.predict_by_feat(scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp)
        if meta_datas[0] is not None: 
            ret['meta_datas'] = meta_datas
            if view_dependents[0] is not None:
                ret['view_dependents'] = view_dependents
        
        scene_loss = self.scene_encoder_criterion(logits, targets)
        ret['scene_loss'] = scene_loss * LOSS_SCENE_OBJ_WEIGHT

        return ret
    
    def predict_by_feat(self, scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp):
        # B is 1 when predecit
        spious, pious, sp_r_ious, p_r_ious, pred_pmasks, scan_idss = [], [], [], [], [], []
        seed_inds, sample_inds, indis = [], [], []
        nt_labels = []
        b = len(object_idss)
        for i in range(b):
            gt_pmask = gt_pmasks[i]
            gt_spmask = gt_spmasks[i]
            pred_spmask = out['masks'][i].squeeze()

            pred_indis = out['indis'][i] # [n_q, 2]
            # take the 1 
            if self.task_type == 'gres':
                indicate = pred_indis.argmax(-1) == 1
                top = indicate.nonzero(as_tuple=False).squeeze(-1)
                # nt_label
                is_nt = False
                if len(top)==0:
                    is_nt = True
                    piou = torch.tensor(0.0, device=pred_spmask.device)
                    spiou = torch.tensor(0.0, device=pred_spmask.device)
                    pred_spmask = torch.zeros_like(pred_spmask[0], device=pred_spmask.device)
                    pred_pmask = pred_spmask[superpoints]

                else:
                    top_mask = pred_spmask[top] # [k, n_sp]
                    pred_spmask = top_mask.max(0)[0] # [n_sp,]
                    spiou = get_iou(pred_spmask, gt_spmask)
                    pred_pmask = pred_spmask[superpoints]
                    piou = get_iou(pred_pmask, gt_pmask)
                    if not is_nt:
                        pred_pmask_binay = pred_pmask.sigmoid() > 0.5
                        is_nt = True if pred_pmask_binay.sum() < 50 else False
                nt_labels.append(is_nt)
            elif self.task_type == 'res':
                softmax_indis = F.softmax(pred_indis, dim=-1)
                indicate = softmax_indis[:,1]
                #print(indicate.max(), (indicate>0.75).sum())
                top = indicate.argmax()

                pred_spmask = pred_spmask[top]
                spiou = get_iou(pred_spmask, gt_spmask)
                pred_pmask = pred_spmask[superpoints]
                piou = get_iou(pred_pmask, gt_pmask)
            else:
                raise NotImplementedError

            spious.append(spiou.cpu())
            pious.append(piou.cpu())
            pred_pmasks.append(pred_pmask.sigmoid().cpu())
            scan_idss.append(scan_ids[0])
            seed_inds.append(fps_seed_sp[i].cpu())
            sample_inds.append(out['sample_inds'][i].cpu())
            indis.append(indicate.cpu())
        gt_pmasks = [gt_pmask.cpu() for gt_pmask in gt_pmasks]
        return dict(scan_id=scan_idss, object_ids=object_idss, ann_id=ann_ids, piou=pious, spiou=spious, gt_pmask=gt_pmasks, pred_pmask=pred_pmasks,
                    sp_r_iou=sp_r_ious, p_r_iou=p_r_ious, nt_label = nt_labels, seed_ind=seed_inds, sample_ind=sample_inds, indi=indis)

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x) # - Initial sparse convolution
        x, mutil_x = self.unet(x) #  - The Sparse 3D U-Net backbone
        x = self.output_layer(x) # Output normalization and ReLU
        x = x.features[v2p_map.long()]  # (B*N, media) # converting from voxel space back to point space
        #print(f"x shape after backbone: {x.shape}")

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
    
    def expand_and_fps(self, sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode=None):
        sp_ins_labels_expand = []
        #feats_2d_expand = feats_2d
        
        batch_offsets_expand = batch_offsets[0:1]
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            if i==0:
                sp_feats_expand = sp_feats[s:e].repeat(scenes_len[i],1)
                sp_coords_float_expand = sp_coords_float[s:e].repeat(scenes_len[i],1)
                feats_2d_expand = feats_2d[s:e].repeat(scenes_len[i],1)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)
            else:
                sp_feats_expand = torch.cat((sp_feats_expand, sp_feats[s:e].repeat(scenes_len[i],1)),dim=0)
                sp_coords_float_expand = torch.cat((sp_coords_float_expand, sp_coords_float[s:e].repeat(scenes_len[i],1)))
                feats_2d_expand = torch.cat((feats_2d_expand, feats_2d[s:e].repeat(scenes_len[i],1)),dim=0)
                fps_seed_sp = FurthestPointSampling.apply(sp_coords_float[s:e].unsqueeze(0), self.fps_num)
                fps_seed_sp_expand = torch.cat((fps_seed_sp_expand, fps_seed_sp.squeeze(0).repeat(scenes_len[i],1)),dim=0)
            for j in range(scenes_len[i]):
                batch_offsets_expand = torch.cat((batch_offsets_expand, batch_offsets_expand[-1:]+batch_offsets[i+1:i+2]-batch_offsets[i:i+1]), dim=0)
                if sp_ins_labels:
                    sp_ins_labels_expand.append(sp_ins_labels[i])
        return sp_feats_expand, sp_coords_float_expand, fps_seed_sp_expand, batch_offsets_expand, sp_ins_labels_expand, feats_2d_expand
    
    def precompute_object_prompt_features(self, train_dataset, val_dataset):
        """
        Precompute CLIP text features for ScanNet ground truth object classes
        """
        print(f"NEW LOSS_SCENE_OBJ_WEIGHT: {LOSS_SCENE_OBJ_WEIGHT}")
        
        model_name = 'openai/clip-vit-large-patch14'
        clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        print(f"\tFROZEN CLIP processor loaded: {model_name}")
        clip_model = CLIPTextModel.from_pretrained(model_name)
        clip_model = clip_model.cuda()
        print(f"\tFROZEN CLIP model loaded: {model_name}")

        global_object_names_scene_inputs = set()

        for item in train_dataset.scene_inputs:
            for sub_item in item:
                global_object_names_scene_inputs.add(sub_item['object_name'])

        for item in val_dataset.scene_inputs:
            for sub_item in item:
                global_object_names_scene_inputs.add(sub_item['object_name'])

        global_object_names_scene_inputs = sorted(global_object_names_scene_inputs)
        print(f"Len of global_object_names_scene_inputs: {len(global_object_names_scene_inputs)}")
        # convert set to list
        object_names_list = list(global_object_names_scene_inputs)
        object_names_list = [name.replace("_", " ") for name in object_names_list]

        print("Using Prompted Ensemble for CLIP Text Features")
        print("Prompts: ", TEMPLATES)
        object_prompt_embeddings = []
        for object_name in object_names_list:
            # Create all prompt variants for this object
            prompts_for_object = [template.format(object_name) for template in TEMPLATES]

            # Tokenize all prompts
            lang_tokens = clip_tokenizer(text=prompts_for_object, return_tensors="pt", padding=True, truncation=True, max_length=77)
            for name in lang_tokens.data:
                lang_tokens.data[name] = lang_tokens.data[name].cuda()

            # Encode with CLIP
            with torch.no_grad():
                clip_outputs = clip_model(**lang_tokens)
                prompt_embeddings = clip_outputs.pooler_output  # shape: [num_templates, 768]

            # Normalize each embedding
            prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)

            # Aggregate prompt embeddings (mean ensemble)
            aggregated_embedding = prompt_embeddings.mean(dim=0)  # shape: [768]
            aggregated_embedding = F.normalize(aggregated_embedding, dim=-1)
                
            object_prompt_embeddings.append(aggregated_embedding)

        # Stack all object features
        sentence_feature = torch.stack(object_prompt_embeddings, dim=0)

        self.object_prompt_features = sentence_feature
        self.scene_encoder.object_prompt_features = sentence_feature

        self.object_vocab = {
            name: sentence_feature[i]
            for i, name in enumerate(object_names_list)
        }

        self.object_index = {name: idx for idx, name in enumerate(object_names_list)}
        self.scene_encoder.object_index = self.object_index
        
        print(f"\tComputed prompted ensambled text features for {len(object_names_list)} ScanNet object classes.")
        print(f"\tPRECOMPUTE FINISHED\n\n")

    def calculate_and_use_pos_weight(self, train_dataset, val_dataset):
        all_object_names = {}
        all_scene_inputs = train_dataset.scene_inputs + val_dataset.scene_inputs
        for item in all_scene_inputs:
            for sub_item in item:
                if sub_item['scene_id'] not in all_object_names:
                    all_object_names[sub_item['scene_id']] = set()
                    all_object_names[sub_item['scene_id']].add(sub_item['object_name'])
                else:
                    all_object_names[sub_item['scene_id']].add(sub_item['object_name'])
        
        # convert the set to lists
        for scene_id in all_object_names:
            all_object_names[scene_id] = list(all_object_names[scene_id])
        # for each object name in the list, replace "_" with " "
        for scene_id in all_object_names:
            all_object_names[scene_id] = [name.replace("_", " ") for name in all_object_names[scene_id]]
        
        # Create a mapping from object name to index
        object_to_idx = self.object_index  # since self.object_index maps names to indices

        num_classes = len(self.object_index )
        total_scenes = len(train_dataset.scene_graphs) + len(val_dataset.scene_graphs)
        total_pos = torch.zeros(num_classes)

        # Count object occurrences per scene
        for name, value in all_object_names.items():
            for obj_name in value:
                if obj_name in object_to_idx:
                    class_idx = object_to_idx[obj_name]
                    total_pos[class_idx] += 1
                else:
                    raise ValueError(f"Object '{obj_name}' not found in object_index. Please check the object names in the dataset.")

        # Avoid division by zero
        epsilon = 1e-6
        total_neg = total_scenes - total_pos
        pos_weight = total_neg / (total_pos + epsilon)

        # Print statistics
        print(f"\n[POS_WEIGHT STATS]")
        print(f"  Shape       : {pos_weight.shape}")
        print(f"  Mean        : {pos_weight.mean().item():.4f}")
        print(f"  Min         : {pos_weight.min().item():.4f}")
        print(f"  Max         : {pos_weight.max().item():.4f}")
        print(f"  Non-zero    : {(total_pos > 0).sum().item()} / {num_classes}")

        #print each object name and its corresponding pos_weight
        print("\n[POS_WEIGHT OBJECT MAPPING]")
        for obj_name, idx in self.object_index.items():
            if total_pos[idx] > 0:
                print(f"  {obj_name:20s} : {pos_weight[idx].item():.4f} (Pos: {total_pos[idx].item()}, Neg: {total_neg[idx].item()})")  
        
        self.pos_weight = pos_weight
        self.scene_encoder_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.cuda())