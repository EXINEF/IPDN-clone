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
from transformers import RobertaModel, RobertaTokenizerFast, AutoModel, CLIPTextModel
from pointnet2.pointnet2_utils import FurthestPointSampling
from ipdn.dataset.mine_configs import TEXT_ENCODER

from .asymmetric_loss import AsymmetricLossOptimized

LOSS_SCENE_OBJ_WEIGHT = 1
USE_ASYMMETRIC_LOSS = False
print(f"USE_ASYMMETRIC_LOSS: {USE_ASYMMETRIC_LOSS}")
print(f"USING POS_WEIGHT: {not USE_ASYMMETRIC_LOSS}")
FREEZE_TEXT_ENCODER_FOR_OBJECTS_EMBEDDINGS = False
AVAILABLES_TEXT_ENCODERS = ['roberta-base',
                'openai/clip-vit-large-patch14', 
                #'google/siglip2-base-patch16-224',
                ]

# print("USING SO_WEIGHT")
# WEIGHT = 20.0
# print(f"WEIGHT: {WEIGHT}")

BOOL_GLOBAL_OBJECT_NAMES_PATH = "/nfs/data_todi/jli/Alessio_works/IPDN-clone/bool_claude_DMA_preprocess_global_object_names__20250507-101601_winsize20_minocc6.json"
PREPROCESSED_FILTERED_OBJECT_IDS_PATH = "/nfs/data_todi/jli/Alessio_works/RAM-clone/output_preprocess_object_ids/neighbor-filtering__20250507-101601_winsize20_minocc6"
SCENE_LIST_PATH = "/nfs/data_todi/jli/Alessio_works/LESS-clone/data/scannet/meta_data/scannetv2-mod.txt"

# BOOL_GLOBAL_OBJECT_NAMES_PATH = "/home/disi/Alessio/IPDN-clone/bool_claude_DMA_preprocess_global_object_names__20250507-101601_winsize20_minocc6.json"
# PREPROCESSED_FILTERED_OBJECT_IDS_PATH = "/home/disi/Alessio/RAM-clone/output_preprocess_object_ids/neighbor-filtering__20250507-101601_winsize20_minocc6"
# SCENE_LIST_PATH = "/home/disi/Alessio/LESS-clone/data/scannet/meta_data/scannetv2-mod.txt"


USE_RANDOM_TEMPLATE = True
TEMPLATES = [
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
]

FEATURE_DIM = 32          # input superpoint dim
HIDDEN_DIM = 256          # internal projected dim
CLIP_DIM = 768            # frozen CLIP embedding dim

class SceneEncoder(nn.Module):
    def __init__(self, object_vocab, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, clip_dim=CLIP_DIM):
        super().__init__()
        # MLP for projecting superpoint features
        self.superpoint_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),      # 32 → 256
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, clip_dim),         # 256 → 768
            nn.LayerNorm(clip_dim),                  # Keep this
        )
        # TODO: we can try to remove the relu
        # project from 32 to 256, then 256 to 768, sp_num x 32 -> sp_num x 768
        # use the mean of the superpoints to get the scene representation, sp_num x 768 -> 768 ,simple average pooling -> scene_feat = sp_proj.mean(dim=0)  # [hidden_dim]
        # normalize the scene representation to get the final embedding 768 -> 768, to avoid gradient explosion

        # Projection to CLIP space
        self.object_vocab = object_vocab

    def forward(self, sp_feats_batch, batch_object_names, clip_matrix):
        """
        sp_feats_batch: list of [num_sp_i, feature_dim] tensors, len = B
        batch_object_names: list of list of object names present in each scene
        clip_text_embeddings: Dict[str, Tensor(768,)], frozen CLIP embeddings
        """
        device = sp_feats_batch[0].device
        B = len(sp_feats_batch)
    
        logits_list = []
        labels_list = []
        
        for i in range(B):
            sp_feats = sp_feats_batch[i]  # [num_sp_i, 32]

            # Project superpoints
            sp_proj = self.superpoint_proj(sp_feats)  # [num_sp_i, hidden_dim]

            # Aggregate scene representation (mean)
            scene_feat = sp_proj.mean(dim=0)  # [hidden_dim]
            # normalize the scene representation
            scene_feat = F.normalize(scene_feat, dim=0)   # [hidden_dim]

            # Similarity to all object embeddings
            clip_matrix_norm = F.normalize(clip_matrix, dim=1)  # [num_objects, 768]
            logits = torch.matmul(clip_matrix_norm, scene_feat)  # [num_objects]

            # Multi-label binary target
            label = torch.zeros(len(self.object_vocab), device=device)
            for obj in batch_object_names[i]:
                if obj in self.object_vocab:
                    label[self.object_vocab[obj]] = 1.0

            # TODO keep 0 for uncorrect objects and 1 for correct objects
            # TODO just modify the weight loss 
            # TODO apply studio label generation and see the accuracy,recall and f1 among all samples
            # fix this code with the new idea.

            logits_list.append(logits)
            labels_list.append(label)

        # Stack for batch loss
        logits_batch = torch.stack(logits_list)  # [B, num_objects]
        labels_batch = torch.stack(labels_list)  # [B, num_objects]

        return logits_batch, labels_batch

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
        if TEXT_ENCODER not in AVAILABLES_TEXT_ENCODERS:
            raise ValueError(f"Invalid TEXT_ENCODER: {TEXT_ENCODER}. Available options: {AVAILABLES_TEXT_ENCODERS}")
        
        if TEXT_ENCODER == 'roberta-base':
            print(f"Using RobertaModel as text encoder: {TEXT_ENCODER}")
            self.text_encoder = RobertaModel.from_pretrained(TEXT_ENCODER)
        
        # if self.architecture_model == 'google/siglip2-base-patch16-224':
        #     print(f"Using Siglip2Model as text encoder: {TEXT_ENCODER_TO_USE}")
        #     self.text_encoder = AutoModel.from_pretrained(TEXT_ENCODER_TO_USE)

        if TEXT_ENCODER == 'openai/clip-vit-large-patch14':
            print(f"Using CLIPTextModel as text encoder: {TEXT_ENCODER}")
            self.text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER)

        self.sampling_module = sampling_module
        self.dec = DEC(**dec, sampling_module=sampling_module, in_channel=media)

        self.object_prompt_features, _ = self.precompute_object_prompt_features()
        
        self.scene_encoder = SceneEncoder(
            object_vocab=self.object_vocab,
            feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            clip_dim=CLIP_DIM
        )
        # criterion
        self.criterion = Criterion(**criterion)
        if USE_ASYMMETRIC_LOSS:
            GAMMA_NEG = 2
            GAMMA_POS = 0
            CLIP = 0.01
            print(f"Using Asymmetric Loss with gamma_neg={GAMMA_NEG}, gamma_pos={GAMMA_POS}, clip={CLIP}")
            self.scene_encoder_criterion = AsymmetricLossOptimized(
                gamma_neg=GAMMA_NEG,
                gamma_pos=GAMMA_POS,  
                clip=CLIP
            )
        else:
            self.scene_encoder_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.cuda())

        # # Test with same inputs
        # logits_test = torch.randn(2, 10, requires_grad=True)  # Example shapes
        # targets_test = torch.randint(0, 2, (2, 10)).float()

        # # Compare BCE and ASL
        # bce_loss = F.binary_cross_entropy_with_logits(logits_test, targets_test)
        # asl_loss = self.scene_encoder_criterion(logits_test, targets_test)

        # print(f"logits values: {logits_test}")
        # print(f"targets values: {targets_test}")
        # print(f"BCE loss: {bce_loss.item():.4f}")
        # print(f"ASL loss: {asl_loss.item():.4f}")
        # exit()

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
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        if not FREEZE_TEXT_ENCODER_FOR_OBJECTS_EMBEDDINGS:
            prompts = self.get_objects_prompts()
            if TEXT_ENCODER == 'roberta-base':
                obj_token_dict = self.roberta_tokenizer.batch_encode_plus(
                    prompts, padding="longest", return_tensors="pt"
                )
                for name in obj_token_dict.data:
                    obj_token_dict.data[name] = obj_token_dict.data[name].cuda()
                obj_lang_tokenss = obj_token_dict['input_ids']
                obj_lang_masks = obj_token_dict['attention_mask']
                self.object_prompt_features = self.text_encoder(obj_lang_tokenss, attention_mask=obj_lang_masks)[0]
                self.object_prompt_features = self.object_prompt_features[:, 0, :] # this is the [CLS] token representation

            if TEXT_ENCODER == 'openai/clip-vit-large-patch14':
                lang_tokens = self.clip_tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
                for name in lang_tokens.data:
                    lang_tokens.data[name] = lang_tokens.data[name].cuda()
                clip_outputs = self.text_encoder(**lang_tokens)
                self.object_prompt_features = clip_outputs.pooler_output

        batch_object_names = self.get_batch_object_names(scan_ids)
        clip_text_embeddings = []
        for i, scene_id in enumerate(scan_ids):
            clip_text_embeddings.append(self.get_scene_object_embeddings(scene_id))

        sp_feats_batch = []
        clip_text_embeddings = {}
        for i in range(len(scenes_len)):
            s = batch_offsets[i]
            e = batch_offsets[i+1]
            sp_feats_batch.append(sp_feats[s:e])
        
        logits, targets = self.scene_encoder(sp_feats_batch, batch_object_names, self.object_prompt_features)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='tr')
        
        if TEXT_ENCODER == 'roberta-base':
            lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]

        if TEXT_ENCODER == 'openai/clip-vit-large-patch14':
            lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)
            lang_feats = lang_feats.last_hidden_state

        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d)
        
        loss, loss_dict = self.criterion(out, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels, dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets)
        
        scene_loss = self.scene_encoder_criterion(logits, targets)
        loss_dict['scene_loss'] = scene_loss * LOSS_SCENE_OBJ_WEIGHT
        loss += scene_loss * LOSS_SCENE_OBJ_WEIGHT
        loss_dict['total_loss'] = loss
        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='pred')
        lang_feats = self.text_encoder(lang_tokenss, attention_mask=lang_masks)[0]
        
        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d) 
        ret = self.predict_by_feat(scan_ids, object_idss, ann_ids, out, superpoints, gt_pmasks, gt_spmasks, fps_seed_sp)
        if meta_datas[0] is not None: 
            ret['meta_datas'] = meta_datas
            if view_dependents[0] is not None:
                ret['view_dependents'] = view_dependents
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
    
    def precompute_object_prompt_features(self):
        """
        Precompute CLIP text features for object prompts
        Args:
            object_classes: List of object class names
            clip_model: CLIP model for text encoding
        Returns:
            Tensor of shape [num_classes, 768]
        """
        print(f"NEW LOSS_SCENE_OBJ_WEIGHT: {LOSS_SCENE_OBJ_WEIGHT}")
        print("\n\nUsing Preprocessed Filtered Object IDs: ", PREPROCESSED_FILTERED_OBJECT_IDS_PATH)

        # print("Using Global Object Names: ", GLOBAL_OBJECT_NAMES_PATH)
        print("\tUsing Random Template: ", USE_RANDOM_TEMPLATE)
        print("\tTemplates: ", TEMPLATES)
        print("\tUsing Bool Global Object Names: ", BOOL_GLOBAL_OBJECT_NAMES_PATH)
        if os.path.exists(BOOL_GLOBAL_OBJECT_NAMES_PATH):
            with open(BOOL_GLOBAL_OBJECT_NAMES_PATH, 'r') as f:
                object_vocab_data  = json.load(f)
        else:
            raise FileNotFoundError(f"Bool Global object names file not found at {BOOL_GLOBAL_OBJECT_NAMES_PATH}")

        filtered_sorted_items = sorted((name for name, value in object_vocab_data.items() if value))
        self.object_vocab = {name: idx for idx, name in enumerate(filtered_sorted_items)}
        print(f"\tLoaded {len(self.object_vocab)} global object names from the file.")
        self.len_object_vocab = len(self.object_vocab)

        # load each line in the file in  a list
        with open(SCENE_LIST_PATH, 'r') as f:
            scene_list = f.readlines()
        scene_list = [x.strip() for x in scene_list]
        scene_list = sorted(scene_list)

        self.object_prompt_names = {}

        for scene_id in scene_list:
            object_names_path = os.path.join(PREPROCESSED_FILTERED_OBJECT_IDS_PATH, scene_id + "_filtered_objects.json")
            if os.path.exists(object_names_path):
                with open(object_names_path, 'r') as f:
                    object_names = json.load(f)
            else:
                raise FileNotFoundError(f"Filtered object IDs file not found for scene {scene_id}")
            # Filter object names to only include those in the object vocabulary
            object_names = {name: value for name, value in object_names.items() if name in self.object_vocab}
            object_names = list(object_names.keys())
            self.object_prompt_names[scene_id] = object_names
        
        print(f"\tLoaded {len(self.object_prompt_names)} scene object names from the file.")

        num_classes = self.len_object_vocab
        total_scenes = len(self.object_prompt_names)
        total_pos = torch.zeros(num_classes)

        # Invert the vocab to map object names to indices
        object_to_idx = self.object_vocab

        for scene_id, object_names in self.object_prompt_names.items():
            for obj_name in object_names:
                class_idx = object_to_idx[obj_name]
                total_pos[class_idx] += 1

        # Avoid division by zero
        epsilon = 1e-6
        total_neg = total_scenes - total_pos
        pos_weight = total_neg / (total_pos + epsilon)
        self.pos_weight = pos_weight

        # Print statistics
        print(f"\n[POS_WEIGHT STATS]")
        print(f"  Shape       : {self.pos_weight.shape}")
        print(f"  Mean        : {self.pos_weight.mean().item():.4f}")
        print(f"  Min         : {self.pos_weight.min().item():.4f}")
        print(f"  Max         : {self.pos_weight.max().item():.4f}")
        print(f"  Non-zero    : {(total_pos > 0).sum().item()} / {num_classes}")

        min_clip = 1.0
        max_clip = 100.0
        self.pos_weight = torch.clamp(pos_weight, min=min_clip, max=max_clip)

        # Print statistics
        print(f"\n[POS_WEIGHT STATS after clamping]")
        print(f"  Shape       : {self.pos_weight.shape}")
        print(f"  Mean        : {self.pos_weight.mean().item():.4f}")
        print(f"  Min         : {self.pos_weight.min().item():.4f}")
        print(f"  Max         : {self.pos_weight.max().item():.4f}")
        print(f"  Non-zero    : {(total_pos > 0).sum().item()} / {num_classes}")        

        if FREEZE_TEXT_ENCODER_FOR_OBJECTS_EMBEDDINGS:
            prompts = self.get_objects_prompts()

            model_name = 'openai/clip-vit-large-patch14'
            clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
            print(f"\tFROZEN CLIP processor loaded: {model_name}")
            clip_model = CLIPTextModel.from_pretrained(model_name)
            clip_model = clip_model.cuda()
            print(f"\tFROZEN CLIP model loaded: {model_name}")

            with torch.no_grad():
                lang_tokens = clip_tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
                for name in lang_tokens.data:
                    lang_tokens.data[name] = lang_tokens.data[name].cuda()
                clip_outputs = clip_model(**lang_tokens)
                sentence_feature = clip_outputs.pooler_output
                attention_mask = lang_tokens.data['attention_mask']
            
            print(f"\tComputed text features for {len(prompts)} object prompts.")
            print(f"\tPRECOMPUTE FINISHED\n\n")
            return sentence_feature, attention_mask
        else:
            if TEXT_ENCODER == 'roberta-base':
                self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained(TEXT_ENCODER)
                print("Loaded Roberta tokenizer for text encoding.")
                print(f"\tNOT using frozen Roberta text encoder for object embeddings, will compute on-the-fly.")
            if TEXT_ENCODER == 'openai/clip-vit-large-patch14':
                self.clip_tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENCODER)
                print("Loaded CLIP tokenizer for text encoding.")
                print(f"\tNOT using frozen CLIP text encoder for object embeddings, will compute on-the-fly.")
            
            return None, None

    def get_objects_prompts(self):
        prompts = []
        for object_name in self.object_vocab.keys():
            if USE_RANDOM_TEMPLATE:
                template = random.choice(TEMPLATES)
            else:
                # Use first template to be consistent
                template = TEMPLATES[0]
            prompt = template.format(object_name)
            prompts.append(prompt)
        return prompts
    
    def get_scene_object_embeddings(self, scene_id):
        """
        Get embeddings for objects in the current scene
        Args:
            scene_id: ID of the current scene
        Returns:
            Tensor of shape [num_scene_objects, 768]
        """
        scene_objects = self.object_prompt_names[scene_id]
        object_indices = [self.object_vocab[obj] for obj in scene_objects]
        return self.object_prompt_features[object_indices]
    
    def get_batch_object_names(self, scan_ids):
        batch_object_names = []
        for scan_id in scan_ids:
            if scan_id in self.object_prompt_names:
                # print(f"Scan ID {scan_id} found in object prompt names.")
                batch_object_names.append(self.object_prompt_names[scan_id])
            else:
                raise ValueError(f"Scan ID {scan_id} not found in object prompt names.")
        return batch_object_names