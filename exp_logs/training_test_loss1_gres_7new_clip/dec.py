import torch
import torch.nn as nn
from .sample_model import SamplingModule
import torch.nn.functional as F
from ..torch.nn import MultiheadAttention

class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask=None, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0, glu = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = glu
        if glu:
            self.glu_p = nn.Linear(d_model, d_model)
            self.glu_g = nn.Linear(d_model, d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None, norm=True):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            # print(f"q shape: {q.shape}, k shape: {k.shape}, x shape: {x.shape}")
            # print(f"x_mask shape: {x_mask.shape if x_mask is not None else None}")
            # print(f"attn_mask before shape: {attn_mask.shape}")
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            # print(f"attn_mask after shape: {attn_mask.shape}")
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        if self.glu:
            output = self.glu_p(output) * (self.glu_g(output).sigmoid())
        output = self.dropout(output) + x
        if norm: output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, norm=True):
        output = self.net(x)
        output = output + x
        if norm: output = self.norm(output)
        return output

# class ObjectCrossAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.0):
#         super().__init__()
#         self.attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, scene_object_embeds, query, scenes_len):
#         """
#         Args:
#             scene_object_embeds: List of tensors [(num_objs1, d_model), (num_objs2, d_model)...]
#             query: Tensor [B_utterances, num_queries, d_model]
#             scenes_len: List indicating number of utterances per scene [5, 6, ...]
#         """
#         device = query.device
#         B_utterances = query.shape[0]
#         d_model = query.shape[-1]

#         # Verify scenes_len matches the total utterances
#         assert sum(scenes_len) == B_utterances, f"Sum of scenes_len {sum(scenes_len)} doesn't match batch size {B_utterances}"
        
#         # Create utterance-to-scene mapping
#         utterance_to_scene = []
#         for scene_idx, num_utterances in enumerate(scenes_len):
#             utterance_to_scene.extend([scene_idx] * num_utterances)
        
#         # Find max objects across all scenes for padding
#         max_objects = max([obj_emb.shape[0] for obj_emb in scene_object_embeds])
        
#         # Create padded tensor for all utterances
#         padded_objs = torch.zeros(B_utterances, max_objects, 768, device=device)
#         attention_mask = torch.ones(B_utterances, max_objects, device=device, dtype=torch.bool)
        
#         # Fill in the padded tensor with appropriate scene objects for each utterance
#         for utterance_idx in range(B_utterances):
#             scene_idx = utterance_to_scene[utterance_idx]
#             obj_emb = scene_object_embeds[scene_idx]
#             num_objs = obj_emb.shape[0]
            
#             padded_objs[utterance_idx, :num_objs] = obj_emb
#             attention_mask[utterance_idx, :num_objs] = False  # False = attend to this position
        
#         # Perform attention with padding mask
#         output, attention_weights, _ = self.attn(
#             query, padded_objs, padded_objs, 
#             key_padding_mask=attention_mask
#         )
        
#         # Apply residual connection and normalization
#         output = self.dropout(output)
#         output = output + query
#         output = self.norm(output)
        
#         return output, attention_weights, None

class HighDimObjectCrossAttention(nn.Module):
    def __init__(self, query_dim=256, obj_dim=768, nhead=8, dropout=0.0, output_projection=None):
        """
        Cross-attention that operates in high dimensions and projects back to model dimension.
        
        Args:
            query_dim: The dimension of the query tensor (typically 256)
            obj_dim: The dimension of the object embeddings (typically 768 for CLIP)
            nhead: Number of attention heads
            dropout: Dropout probability
            output_projection: Shared projection layer from obj_dim to query_dim
        """
        super().__init__()
        
        # Ensure head_dim is divisible by nhead
        assert obj_dim % nhead == 0, f"obj_dim {obj_dim} must be divisible by nhead {nhead}"
        
        # Projection layers
        self.query_proj = nn.Linear(query_dim, obj_dim)
        self.key_proj = nn.Linear(obj_dim, obj_dim)
        self.value_proj = nn.Linear(obj_dim, obj_dim)
        
        self.output_projection = output_projection
   
        # Multi-head attention parameters
        self.nhead = nhead
        self.head_dim = obj_dim // nhead
        self.scale = self.head_dim ** -0.5
        
        # Normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(query_dim)
        
    def _reshape_for_multihead(self, x, batch_size, seq_len):
        """Reshape input tensor for multi-head attention"""
        return x.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
    def forward(self, scene_object_embeds, query, scenes_len):
        """
        Args:
            scene_object_embeds: List of tensors [(num_objs1, 768), (num_objs2, 768)...]
            query: Tensor [B_utterances, num_queries, 256]
            scenes_len: List indicating number of utterances per scene
        """
        device = query.device
        batch_size = query.shape[0]
        n_queries = query.shape[1]
        # num_scenes = len(scene_object_embeds)
        
        # Create utterance-to-scene mapping
        utterance_to_scene = []
        for scene_idx, num_utterances in enumerate(scenes_len):
            utterance_to_scene.extend([scene_idx] * num_utterances)
        
        # Find max objects across all scenes for padding
        max_objects = max([obj_emb.shape[0] for obj_emb in scene_object_embeds])
        
        # Create padded tensor for all utterances (using original CLIP dimension)
        padded_objs = torch.zeros(batch_size, max_objects, scene_object_embeds[0].shape[-1], device=device)
        attention_mask = torch.ones(batch_size, max_objects, device=device, dtype=torch.bool)
        
        # Fill in the padded tensor with appropriate scene objects
        for utterance_idx in range(batch_size):
            scene_idx = utterance_to_scene[utterance_idx]
            obj_emb = scene_object_embeds[scene_idx]
            num_objs = obj_emb.shape[0]
            
            padded_objs[utterance_idx, :num_objs] = obj_emb
            attention_mask[utterance_idx, :num_objs] = False  # False = attend to this position
        
        # Project query, key, value to high-dimensional space
        q = self.query_proj(query)  # [batch_size, n_queries, obj_dim]
        k = self.key_proj(padded_objs)  # [batch_size, max_objects, obj_dim]
        v = self.value_proj(padded_objs)  # [batch_size, max_objects, obj_dim]
        
        # Reshape for multi-head attention
        q = self._reshape_for_multihead(q, batch_size, n_queries)  # [batch_size, nhead, n_queries, head_dim]
        k = self._reshape_for_multihead(k, batch_size, max_objects)  # [batch_size, nhead, max_objects, head_dim]
        v = self._reshape_for_multihead(v, batch_size, max_objects)  # [batch_size, nhead, max_objects, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, nhead, n_queries, max_objects]
        
        # Apply mask to prevent attention to padding
        if attention_mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_objects]
            attn_scores = attn_scores.masked_fill(expanded_mask, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, nhead, n_queries, max_objects]
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [batch_size, nhead, n_queries, head_dim]
        
        # Reshape back to regular tensor format
        context = context.transpose(1, 2).contiguous().view(batch_size, n_queries, -1)  # [batch_size, n_queries, obj_dim]
        
        # Project back to query dimension using shared projection
        output = self.output_projection(context)  # [batch_size, n_queries, query_dim]
        
        # Apply residual connection and normalization
        output = self.dropout(output)
        output = output + query
        output = self.norm(output)
        
        return output, attn_weights, None  # Match return signature of other attention layers
    
class DEC(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """
    def __init__(
        self,
        num_layer=6,
        num_class=256,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        sampling_module=None,
        kernel='top1',
        global_feat='mean',
        lang_att=False,
        contrastive_align_loss=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_class = num_class
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.input_proj_2d = nn.Sequential(nn.Linear(1024, d_model*2),nn.ReLU(),nn.Linear(d_model*2, d_model),nn.ReLU(),nn.Linear(d_model, d_model))#clip
        self.sum_norm = nn.LayerNorm(d_model)
        
        self.lang_att = lang_att
        self.contrastive_align_loss = contrastive_align_loss

        H = 768
        self.lang_proj = nn.Linear(H, d_model)
        self.lang_norm = nn.LayerNorm(d_model)
        
        if sampling_module is not None:
            self.sampling_module = SamplingModule(**sampling_module)
        else:
            self.sampling_module = None

        self.query_generator = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model))
        
        self.object_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)  # d_model is 256
        )

        # DDI and SWA
        self.swa_layers = nn.ModuleList([])
        self.rra_layers = nn.ModuleList([])
        self.rla_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        self.sem_cls_heads = nn.ModuleList([])
        self.scg = nn.ModuleList([])
        self.lqg = nn.ModuleList([])
        self.obj_att_layers = nn.ModuleList([])
        if self.lang_att:
            self.lla_layers = nn.ModuleList([])
            self.lsa_layers = nn.ModuleList([])
            self.lsa_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.swa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.rra_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.rla_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.sem_cls_heads.append(ThreeLayerMLP(d_model, self.num_class))
            self.scg.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.lqg.append(CrossAttentionLayer(d_model, nhead, dropout))
            if self.lang_att:
                self.lla_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
                self.lsa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
                self.lsa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))    
            self.obj_att_layers.append(
                HighDimObjectCrossAttention(
                    query_dim=d_model, 
                    obj_dim=768, 
                    nhead=nhead, 
                    dropout=dropout,
                    output_projection=self.object_projection # Pass shared projection
                )
            )

        self.sem_cls_head = ThreeLayerMLP(d_model, self.num_class)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
        self.indi_embedding = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2), nn.Linear(2, 2))
        self.indi_norm = nn.LayerNorm(d_model)

        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.kernel = kernel
        self.global_feat = global_feat
        
        self.k = 8

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_vision = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, 64))
            self.contrastive_align_projection_text = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, 64))         
        
        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)
        
        
    def forward_iter_pred(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None, sp_pos=None, feats_2d=None, scene_object_embeds=None, scenes_len=None):
        """
        x [B*M, inchannel]
        """
        
        lang_feats = self.lang_proj(lang_feats) 
        lang_feats = self.lang_norm(lang_feats)  
        lang_masks = ~(lang_masks.bool())
        inst_feats = self.input_proj(x)
        
        inst_feats_2d = self.input_proj_2d(feats_2d)
        inst_feats = self.sum_norm(inst_feats+inst_feats_2d)
        
        mask_feats = self.x_mask(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        
        sp_pos, _ = self.get_batches(sp_pos, batch_offsets)
        
        prediction_masks, prediction_scores, prediction_classes, prediction_indis = [], [], [], []
        
        sample_inds = None
        ref_scores = None
         
        seed_sp = inst_feats.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, inst_feats.size(-1)))
        seed_sp_pos = sp_pos.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, sp_pos.size(-1)))
        
        scg_mask = self.get_scg_mask(seed_sp_pos)
        seed_sp = self.scg_head(seed_sp, attn_mask = ~scg_mask)

         # sampling
        if hasattr(self, 'sampling_module') and self.sampling_module is not None:
            sample_inds, ref_scores = self.sampling_module(seed_sp, lang_feats, None, lang_masks)
            sample_inds = sample_inds.long()
            sampled_seed = seed_sp.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp.size(-1)))
            query_pos = seed_sp_pos.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp_pos.size(-1)))
            query = self.query_generator(sampled_seed)
        else:
            query = self.query_generator(seed_sp)
            query_pos = seed_sp_pos
            
        scg_mask = self.get_scg_mask(query_pos)

        proj_queries = []
        if self.contrastive_align_loss:
            proj_queries.append(F.normalize(self.contrastive_align_projection_vision(query), p=2, dim=-1))
        else:
            proj_queries.append(None)
            
        proj_tokens = []
        if self.contrastive_align_loss:
            proj_tokens.append(F.normalize(self.contrastive_align_projection_text(lang_feats), p=2, dim=-1))
        else:
            proj_tokens.append(None)
        
        
        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
        pred_indis = self.indi_embedding(query)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)
        prediction_indis.append(pred_indis)
        
        init_lang_mask = lang_masks
        #lang_query = lang_feats
        
        # if scene_object_embeds is not None:
        #     projected_scene_embeds = []
        #     for scene_embeds in scene_object_embeds:
        #         projected_scene_embeds.append(self.object_projection(scene_embeds))

        # multi-round
        l = 0
        for i in range(self.num_layer):
            #PAD
            lang_query, w1, w2 = self.lqg[i](query[:,:128],lang_feats)
            if i > l:
                mask = pred_indis.softmax(-1)[:,:,1]>0.75
                weight = (w2 * (~init_lang_mask).unsqueeze(-1)).sum(1)
                weight = torch.masked_fill(weight,~mask,float('-inf')).softmax(-1)
                weight = torch.masked_fill(weight, torch.isnan(weight), 0)
                #prompt = torch.einsum('abc,acd->abd',weight.unsqueeze(1),query[:,:128])
                prompt = weight.unsqueeze(-1) * (query[:,:128])
                #prompt = self.prompt[i].unsqueeze(0).repeat(query.shape[0],1,1)
                query = torch.cat([query[:,:128], prompt], dim=1)
                _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
            
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](inst_feats, lang_query, batch_mask, None)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            query, _, _ = self.swa_layers[i](inst_feats, query, batch_mask, attn_masks)
            query_rra = self.rra_layers[i](query)
            query_rla, _, _ = self.rla_layers[i](lang_query, query, lang_masks)
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](query, lang_query)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            if scene_object_embeds is not None:
                query_obj, _, _ = self.obj_att_layers[i](scene_object_embeds, query, scenes_len)
                query = query + query_rla + query_obj + query_rra
            else:
                query = query + query_rla + query_rra
                
            if i <= l:
                query = self.scg[i](query, attn_mask=~scg_mask)
            else:
                query = torch.cat([self.scg[i](query[:,:128], attn_mask=~scg_mask),query[:,128:]],dim=1)
            
            query = self.swa_ffn_layers[i](query)


            pred_scores, pred_masks, attn_masks = self.prediction_head(query[:,:128], mask_feats, batch_mask)
            pred_indis = self.indi_embedding(query[:,:128])
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
            prediction_indis.append(pred_indis)
            if self.contrastive_align_loss:
                proj_queries.append(F.normalize(self.contrastive_align_projection_vision(query[:,:128]), p=2, dim=-1))
            else:
                proj_queries.append(None)
                
            if self.contrastive_align_loss:
                proj_tokens.append(F.normalize(self.contrastive_align_projection_text(lang_query), p=2, dim=-1))
            else:
                proj_tokens.append(None)

        return {
            'masks': pred_masks,
            'batch_mask': batch_mask,
            'scores': pred_scores,
            'indis': pred_indis, # [B, B_q, 2]
            'proj_queries': proj_queries[-1],
            'proj_tokens': proj_tokens[-1],
            'query_features': query[:,:128], # [B, B_q, d_model]
            'sample_inds': sample_inds, # [B, K]
            'ref_scores': ref_scores, # [B, M]
            'aux_outputs': [{
                'masks': a,
                'scores': b,
                'proj_queries': c,
                'indis': d,
                'proj_tokens': e,
            } for a, b, c, d, e in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
                proj_queries[:-1],
                prediction_indis[:-1],
                proj_tokens[:-1],
            )],
        }
        
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks

    def get_scg_mask(self, pos_q, pos_k=None, k_mask=None, k=8):
        if pos_k == None: pos_k = pos_q
        scg_mask = torch.zeros((pos_q.shape[0],pos_q.shape[1],pos_k.shape[1]),device=pos_q.device).bool()
        dis = ((pos_q.unsqueeze(2)-pos_k.unsqueeze(1))**2).sum(-1)
        if k_mask is not None:
            dis = torch.masked_fill(dis, k_mask.unsqueeze(1).repeat(1,pos_q.shape[1],1), 1000000.0)
        ind = torch.topk(dis, k, dim=-1, largest=False)[1]
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                scg_mask[i,j,ind[i][j]] = True
        return scg_mask
    
    def forward(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None, sp_pos=None, feats_2d=None, scene_object_embeds=None, scenes_len=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_pos, feats_2d, scene_object_embeds, scenes_len)
        else:
            raise NotImplementedError
