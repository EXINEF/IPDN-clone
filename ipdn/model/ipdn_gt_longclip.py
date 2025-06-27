import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools

from ipdn.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou

from .dec_original import DEC
from pointnet2.pointnet2_utils import FurthestPointSampling

import sys
sys.path.append("/nfs/data_todi/jli/Alessio_works/Long-CLIP")
from model import longclip


# LOSS_SCENE_OBJ_WEIGHT = 1
# USE_ASYMMETRIC_LOSS = False
# GAMMA_NEG = 4
# GAMMA_POS = 0
# CLIP = 0.05
# print(f"USE_ASYMMETRIC_LOSS: {USE_ASYMMETRIC_LOSS}")
# print(f"USING POS_WEIGHT: {not USE_ASYMMETRIC_LOSS}")

# # print("USING SO_WEIGHT")
# # WEIGHT = 20.0
# # print(f"WEIGHT: {WEIGHT}")

# # SCENE_LIST_PATH = "/nfs/data_todi/jli/Alessio_works/LESS-clone/data/scannet/meta_data/scannetv2-mod.txt"

# SCENE_LIST_PATH = "/home/disi/Alessio/LESS-clone/data/scannet/meta_data/scannetv2-mod.txt"


# # # USE_RANDOM_TEMPLATE = False
# # # TEMPLATES = [
# # #     "There is a {} in the scene",
# # #     "A scene with {}",
# # #     "A scene containing {}",
# # #     "A room with {}",
# # #     "A room containing {}",
# # #     "There is {} in the scene",
# # #     "There is {} in the room",
# # #     "{} is present in the scene",
# # #     "{} is present in the room",
# # #     "In the scene there is a {}",
# # #     "In the room there is a {}",
# # #     "The scene contains {}",
# # #     "The room contains {}",
# # # ]

# TEMPLATES = [
#     # Original templates
#     "There is a {} in the scene",
#     "A scene with {}",
#     "A scene containing {}",
#     "A room with {}",
#     "A room containing {}",
#     "There is {} in the scene",
#     "There is {} in the room",
#     "{} is present in the scene",
#     "{} is present in the room",
#     "In the scene there is a {}",
#     "In the room there is a {}",
#     "The scene contains {}",
#     "The room contains {}",
#     # Additional templates for better ensemble
#     "A photo of a {} in a room",
#     "A 3D scan of a room with {}",
#     "Indoor scene with {}",
#     "This room has a {}",
#     "You can see a {} here",
#     "A {} can be found in this room",
#     "Look, there's a {} in the scene",
# ]

# FEATURE_DIM = 32          # input superpoint dim
# HIDDEN_DIM = 256          # internal projected dim
# CLIP_DIM = 768            # frozen CLIP embedding dim

# class SceneEncoder(nn.Module):
#     def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, clip_dim=CLIP_DIM):
#         super().__init__()
#         # MLP for projecting superpoint features
#         print("RELU ACTIVATED")
#         self.superpoint_proj = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),      # 32 → 256
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, clip_dim),         # 256 → 768
#             nn.LayerNorm(clip_dim),                  # Keep this
#         )
#         # TODO: we can try to remove the relu
#         # project from 32 to 256, then 256 to 768, sp_num x 32 -> sp_num x 768
#         # use the mean of the superpoints to get the scene representation, sp_num x 768 -> 768 ,simple average pooling -> scene_feat = sp_proj.mean(dim=0)  # [hidden_dim]
#         # normalize the scene representation to get the final embedding 768 -> 768, to avoid gradient explosion

#         # self.object_vocab = object_vocab

#         # self.attn_pool = nn.Sequential(
#         #     nn.Linear(768, 1)
#         # )
#         # self.attn_temperature = 1.0

#     def forward(self, sp_feats_batch, batch_object_names, clip_matrix, object_index):
#         """
#         sp_feats_batch: list of [num_sp_i, feature_dim] tensors, len = B
#         batch_object_names: list of list of object names present in each scene
#         clip_text_embeddings: Dict[str, Tensor(768,)], frozen CLIP embeddings
#         """
#         device = sp_feats_batch[0].device
#         B = len(sp_feats_batch)
    
#         logits_list = []
#         labels_list = []

#         for i in range(B):
#             sp_feats = sp_feats_batch[i]  # [num_sp_i, 32]

#             # Project superpoints
#             sp_proj = self.superpoint_proj(sp_feats)  # [num_sp_i, 768]

#             # Aggregate scene representation (mean)
#             scene_feat = sp_proj.mean(dim=0)  # [768]

#             # # Aggregate scene representation (attention pooling)
#             # attn_weights = self.attn_pool(sp_proj).squeeze(-1)  # [num_sp_i]
#             # attn_weights = F.softmax(attn_weights / self.attn_temperature, dim=0)  # [num_sp_i]
#             # scene_feat = torch.sum(attn_weights.unsqueeze(1) * sp_proj, dim=0)  # [768]

#             # normalize the scene representation
#             #scene_feat = F.normalize(scene_feat, dim=0)   # [768]

#             # Similarity to all object embeddings
#             logits = torch.matmul(clip_matrix, scene_feat)  # [num_objects]

#             # Multi-label binary target
#             label = torch.zeros(len(object_index), device=device, dtype=torch.float32)
#             for obj in batch_object_names[i]:
#                 if obj in object_index:
#                     label[object_index[obj]] = 1.0
#                 else:
#                     raise ValueError(f"Object '{obj}' not found in object_index. Please check the object names in the dataset.")
                
#             logits_list.append(logits)
#             labels_list.append(label)

#         # Stack for batch loss
#         logits_batch = torch.stack(logits_list)  # [B, num_objects]
#         labels_batch = torch.stack(labels_list)  # [B, num_objects]

#         return logits_batch, labels_batch

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

        self.text_encoder, _ = longclip.load("/nfs/data_todi/jli/Alessio_works/Long-CLIP/checkpoints/longclip-B.pt", device="cuda")
        self.text_encoder.float()
        # self.text_encoder, _ = longclip.load("/home/disi/Alessio/Long-CLIP/checkpoints/longclip-B.pt", device="cuda")

        self.sampling_module = sampling_module
        self.dec = DEC(**dec, sampling_module=sampling_module, in_channel=media)

        # self.scene_encoder = SceneEncoder(
        #     feature_dim=FEATURE_DIM,
        #     hidden_dim=HIDDEN_DIM,
        #     clip_dim=CLIP_DIM
        # )
        # criterion
        self.criterion = Criterion(**criterion)
        # if USE_ASYMMETRIC_LOSS:
        #     print(f"Using Asymmetric Loss with gamma_neg={GAMMA_NEG}, gamma_pos={GAMMA_POS}, clip={CLIP}")
        #     self.scene_encoder_criterion = AsymmetricLossOptimized(
        #         gamma_neg=GAMMA_NEG,
        #         gamma_pos=GAMMA_POS,  
        #         clip=CLIP
        #     )
        # else:
        #     self.scene_encoder_criterion = nn.BCEWithLogitsLoss()

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
    def loss(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, gt_objects_names=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='tr')
        
        lang_feats, _ = self.get_clip_token_features(lang_tokenss)

        out = self.dec(sp_feats, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_coords_float, feats_2d)
        
        loss, loss_dict = self.criterion(out, gt_spmasks, sp_ref_masks, object_idss, sp_ins_labels, dense_maps, lang_masks, fps_seed_sp, sp_coords_float, batch_offsets)

        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, object_idss, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, coords_float, sp_ins_labels, dense_maps, scenes_len=None, meta_datas=None, view_dependents=None, feats_2d=None, semantic_labels=None, instance_labels=None,  gt_objects_names=None):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        
        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        sp_coords_float = scatter_mean(coords_float, superpoints, dim=0)

        sp_feats, sp_coords_float, fps_seed_sp, batch_offsets, sp_ins_labels, feats_2d = self.expand_and_fps(sp_feats, sp_coords_float, batch_offsets, sp_ins_labels, scenes_len, feats_2d, mode='pred')
        lang_feats, _ = self.get_clip_token_features(lang_tokenss)
        
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
    

    def get_clip_token_features(self, text_tokens):
        """
        Extract contextualized token features from CLIP's transformer.
        Returns both token features and sentence feature (EOS token).
        """
        # This is similar to CLIP's encode_text but extracts all token representations
        x = self.text_encoder.token_embedding(text_tokens).type(self.text_encoder.dtype)
        x = x + self.text_encoder.positional_embedding.type(self.text_encoder.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_encoder.ln_final(x).type(self.text_encoder.dtype)
        
        # Get sentence feature (from EOS token in CLIP)
        eot_indices = text_tokens.argmax(dim=-1)
        sentence_feature = x[torch.arange(x.shape[0]), eot_indices] @ self.text_encoder.text_projection
        
        # Apply L2 normalization to sentence feature, like original CLIP
        sentence_feature = sentence_feature / sentence_feature.norm(dim=-1, keepdim=True)
        
        # Return all token features and the sentence feature
        return x, sentence_feature
    
    # def precompute_object_prompt_features(self, train_dataset, val_dataset):
    #     """
    #     Precompute CLIP text features for ScanNet ground truth object classes
    #     """
    #     print(f"NEW LOSS_SCENE_OBJ_WEIGHT: {LOSS_SCENE_OBJ_WEIGHT}")
        
    #     model_name = 'openai/clip-vit-large-patch14'
    #     clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
    #     print(f"\tFROZEN CLIP processor loaded: {model_name}")
    #     clip_model = CLIPTextModel.from_pretrained(model_name)
    #     clip_model = clip_model.cuda()
    #     print(f"\tFROZEN CLIP model loaded: {model_name}")

    #     global_object_names_scene_inputs = set()
    #     for item in train_dataset.scene_inputs:
    #         for sub_item in item:
    #             global_object_names_scene_inputs.add(sub_item['object_name'])

    #     for item in val_dataset.scene_inputs:
    #         for sub_item in item:
    #             global_object_names_scene_inputs.add(sub_item['object_name'])

    #     # # print descriptions statistics
    #     # print(f"Total descriptions: {len(all_descriptions)}")
    #     # print(f"Unique descriptions: {len(set(all_descriptions))}")
    #     # # print length statistics
    #     # lengths = [len(desc.split()) for desc in all_descriptions]
    #     # print(f"Description lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}")

    #     # # count how many are 77> length
    #     # long_descriptions = [desc for desc in all_descriptions if len(desc.split()) > 77]
    #     # print(f"Descriptions longer than 77 tokens: {len(long_descriptions)}")
        
    #     # # extract descriptions with lenght of 76 
    #     # descriptions_76 = [desc for desc in all_descriptions if len(desc.split()) == 76]
    #     # print(f"Descriptions with length of 76 tokens: {len(descriptions_76)}")

    #     # longest_descriptions = sorted(all_descriptions, key=lambda x: len(x.split()), reverse=True)[:10]
    #     # print("10 longest descriptions:")
    #     # for desc in longest_descriptions:
    #     #     print(f"\t{desc} (length: {len(desc.split())})")
        
    #     # # tokenize 10 longest descriptions with CLIP tokenizer
    #     # print("Tokenizing 10 longest descriptions with CLIP tokenizer:")
    #     # for desc in descriptions_76:
    #     #     tokens = clip_tokenizer(desc, return_tensors="pt", padding=True)
    #     #     print(f"\t{desc} -> {tokens['input_ids'].squeeze().tolist()} (length: {len(tokens['input_ids'].squeeze())})")


    #     # counter = 0
    #     # for i, desc in enumerate(all_descriptions):
    #     #     token_ids = clip_tokenizer.encode(desc, add_special_tokens=True)
    #     #     token_count = len(token_ids)
    #     #     if token_count > 77:
    #     #         print(f"[{i}] Token count: {token_count} > 77")
    #     #         print(f"Description: {desc}")
    #     #         print()
    #     #         counter += 1
    #     # print(f"Total descriptions with token count > 77: {counter}")

    #     # convert set to list
    #     object_names_list = list(global_object_names_scene_inputs)
    #     object_names_list = [name.replace("_", " ") for name in global_object_names_scene_inputs]

    #     # # # prompts = []
    #     # # # for object_name in object_names_list:
    #     # # #     if USE_RANDOM_TEMPLATE:
    #     # # #         template = random.choice(TEMPLATES)
    #     # # #     else:
    #     # # #         # Use first template to be consistent
    #     # # #         template = TEMPLATES[0]
    #     # # #     prompt = template.format(object_name)
    #     # # #     prompts.append(prompt)

    #     # # # with torch.no_grad():
    #     # # #     lang_tokens = clip_tokenizer(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    #     # # #     for name in lang_tokens.data:
    #     # # #         lang_tokens.data[name] = lang_tokens.data[name].cuda()
    #     # # #     clip_outputs = clip_model(**lang_tokens)
    #     # # #     sentence_feature = clip_outputs.pooler_output
    #     # # #     #attention_mask = lang_tokens.data['attention_mask']
        
    #     # # # # normalize the sentence features
    #     # # # sentence_feature = F.normalize(sentence_feature, dim=-1)

    #     print("Using Prompted Ensemble for CLIP Text Features")
    #     print("Prompts: ", TEMPLATES)
    #     object_prompt_embeddings = []
    #     for object_name in object_names_list:
    #         # Create all prompt variants for this object
    #         prompts_for_object = [template.format(object_name) for template in TEMPLATES]

    #         # Tokenize all prompts
    #         lang_tokens = clip_tokenizer(text=prompts_for_object, return_tensors="pt", padding=True, truncation=True, max_length=77)
    #         for name in lang_tokens.data:
    #             lang_tokens.data[name] = lang_tokens.data[name].cuda()

    #         # Encode with CLIP
    #         with torch.no_grad():
    #             clip_outputs = clip_model(**lang_tokens)
    #             prompt_embeddings = clip_outputs.pooler_output  # shape: [num_templates, 768]

    #         # Normalize each embedding
    #         prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)

    #         # Aggregate prompt embeddings (mean ensemble)
    #         aggregated_embedding = prompt_embeddings.mean(dim=0)  # shape: [768]
    #         aggregated_embedding = F.normalize(aggregated_embedding, dim=-1)
                
    #         object_prompt_embeddings.append(aggregated_embedding)

    #     # Stack all object features
    #     sentence_feature = torch.stack(object_prompt_embeddings, dim=0)
    #     sentence_feature = F.normalize(sentence_feature, dim=-1)

    #     sentence_feature = sentence_feature.detach()  # Detach to avoid gradients
    #     self.object_prompt_features = sentence_feature

    #     self.object_vocab = {
    #         name: sentence_feature[i]
    #         for i, name in enumerate(object_names_list)
    #     }

    #     self.object_index = {name: idx for idx, name in enumerate(object_names_list)}
        
    #     print(f"\tComputed prompted ensambled text features for {len(object_names_list)} ScanNet object classes.")
    #     print(f"\tPRECOMPUTE FINISHED\n\n")

    # def calculate_pos_weight(self, train_dataset, val_dataset):
        
    #     all_object_names = {}
    #     all_scene_inputs = train_dataset.scene_inputs + val_dataset.scene_inputs
    #     for item in all_scene_inputs:
    #         for sub_item in item:
    #             if sub_item['scene_id'] not in all_object_names:
    #                 all_object_names[sub_item['scene_id']] = set()
    #                 all_object_names[sub_item['scene_id']].add(sub_item['object_name'])
    #             else:
    #                 all_object_names[sub_item['scene_id']].add(sub_item['object_name'])
        
    #     # convert the set to lists
    #     for scene_id in all_object_names:
    #         all_object_names[scene_id] = list(all_object_names[scene_id])
    #     # for each object name in the list, replace "_" with " "
    #     for scene_id in all_object_names:
    #         all_object_names[scene_id] = [name.replace("_", " ") for name in all_object_names[scene_id]]
        
    #     # Create a mapping from object name to index
    #     object_to_idx = self.object_index  # since self.object_index maps names to indices

    #     num_classes = len(self.object_index )
    #     total_scenes = len(train_dataset.scene_graphs) + len(val_dataset.scene_graphs)
    #     total_pos = torch.zeros(num_classes)

    #     # Count object occurrences per scene
    #     for name, value in all_object_names.items():
    #         for obj_name in value:
    #             if obj_name in object_to_idx:
    #                 class_idx = object_to_idx[obj_name]
    #                 total_pos[class_idx] += 1
    #             else:
    #                 raise ValueError(f"Object '{obj_name}' not found in object_index. Please check the object names in the dataset.")

    #     # Avoid division by zero
    #     epsilon = 1e-6
    #     total_neg = total_scenes - total_pos
    #     pos_weight = total_neg / (total_pos + epsilon)

    #     # Print statistics
    #     print(f"\n[POS_WEIGHT STATS]")
    #     print(f"  Shape       : {pos_weight.shape}")
    #     print(f"  Mean        : {pos_weight.mean().item():.4f}")
    #     print(f"  Min         : {pos_weight.min().item():.4f}")
    #     print(f"  Max         : {pos_weight.max().item():.4f}")
    #     print(f"  Non-zero    : {(total_pos > 0).sum().item()} / {num_classes}")

    #     #print each object name and its corresponding pos_weight
    #     print("\n[POS_WEIGHT OBJECT MAPPING]")
    #     for obj_name, idx in self.object_index.items():
    #         if total_pos[idx] > 0:
    #             print(f"  {obj_name:20s} : {pos_weight[idx].item():.4f} (Pos: {total_pos[idx].item()}, Neg: {total_neg[idx].item()})")  
        
    #     self.pos_weight = pos_weight
    #     self.scene_encoder_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.cuda())
    #     # # Clip pos_weight
    #     # min_clip = 1.0
    #     # max_clip = 100.0
    #     # pos_weight = torch.clamp(pos_weight, min=min_clip, max=max_clip)

    #     # self.pos_weight = pos_weight

    #     # # Print statistics
    #     # print(f"\n[POS_WEIGHT STATS]")
    #     # print(f"  Shape       : {pos_weight.shape}")
    #     # print(f"  Mean        : {pos_weight.mean().item():.4f}")
    #     # print(f"  Min         : {pos_weight.min().item():.4f}")
    #     # print(f"  Max         : {pos_weight.max().item():.4f}")
    #     # print(f"  Non-zero    : {(total_pos > 0).sum().item()} / {num_classes}")
