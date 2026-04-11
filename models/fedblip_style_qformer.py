# abnblip_qformer_style_CR_EMA_local.py
# cross round style Contrastive
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from transformers import BertConfig, BertLMHeadModel  # 保留，防止底层 QFormer 依赖

from models.base_model import all_gather_with_grad, concat_all_gather
from models.blip2_base_style import Blip2Base
# from fed_server_autoloop_CR import SERVER_MOMENTUM as LOCAL_MOMENTUM 
STYLE_MARGIN = 0.5
STYLE_LAMBDA = 0.5
# STYLE_MARGIN = 0.1
# STYLE_LAMBDA = 0.2
LOCAL_MOMENTUM = 0.95
    
# ========= MLP Head =========
class MLPHead(nn.Module):
    """
    两层 MLP + LayerNorm + Dropout；可选分类器头
    支持 [B,C] 或 [B,N,C] 输入（保持最后一维为通道）
    """
    def __init__(self, in_channels, hidden_dim, dropout=0.1, out_relu=False, num_classes=None):
        super().__init__()
        self.out_relu = out_relu
        self.has_classifier = num_classes is not None

        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        if self.has_classifier:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            if self.out_relu:
                self.norm2 = nn.LayerNorm(hidden_dim)
                # self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop1(x)

        if self.has_classifier:
            x = self.classifier(x)  # [B, ... , num_classes]
            return x
        else:
            x = self.fc2(x)
            if self.out_relu:
                x = self.norm2(x)
                x = self.act(x)
                # x = self.drop2(x)
            return x
        
class Blip2Qformer(Blip2Base):
    """
    Simplified and fixed Abn-BLIP QFormer model.
    """

    def __init__(
        self,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=40,
        abn_num=56,
        client_id=0,
        num_clients=3,
        style_learn=False,
        **kwargs,  # 兼容 from_config 等多余参数，直接忽略
    ):
        super().__init__()
        self.client_id = client_id
        self.num_clients = num_clients
        self.tokenizer = self.init_tokenizer()
        self.abn_size = abn_num
        self.num_query_token = 1
        self.visual_encoder_num_features = 1408

        # ✅ Safe initialization of Q-Former
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.visual_encoder_num_features, cross_attention_freq, abn_num
        )

        # initialize cross-attention layers with self-attention paramters
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                if key_orig in state_dict:
                    param.data.copy_(state_dict[key_orig])

        # ✅ Projection layers
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.hidden_dim = self.query_tokens.shape[-1]

        self.style_learn = style_learn
        if self.style_learn:
            # ✅ style tokens per client (learnable)
            self.n_style_token = 4 # exp 7
            # self.n_style_token = 8
            # self.n_style_token = 10
            self.style_tokens = nn.Parameter(
                torch.randn(self.num_clients, self.n_style_token, self.hidden_dim) * 0.02,
                requires_grad=True,
            )

            D = 64  # style embedding dim
            self.text_style_proj = nn.Linear(self.Qformer.config.hidden_size, D)

            self.register_buffer(
                "style_center",
                torch.zeros(self.num_clients, D)  # 每个client一个风格中心
            )
            self.style_momentum = LOCAL_MOMENTUM  # EMA 系数
        else:
            self.n_style_token = 0

        # ✅ CNN-based feature encoding (5-scale)
        self.layer1 = nn.Sequential(
            nn.Conv3d(64, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 1408, 1, 1, 0),
            nn.Flatten(2, 4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(256, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 1408, 1, 1, 0),
            nn.Flatten(2, 4),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(512, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 1408, 1, 1, 0),
            nn.Flatten(2, 4),
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(1024, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 1408, 1, 1, 0),
            nn.Flatten(2, 4),
            nn.Linear(2 * 2 * 2, self.abn_size),
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(2048, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 1408, 1, 1, 0),
            nn.Flatten(2, 4),
            nn.Linear(7 * 7 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, self.abn_size),
        )

        # ✅ classification branch
        self.cls_embedder = nn.Linear(self.abn_size, 1408)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.classifier = nn.Conv3d(2048, self.abn_size, kernel_size=1)
        self.classifier = MLPHead(2048, embed_dim, num_classes=self.abn_size)

        self.cls_loss_function = nn.BCEWithLogitsLoss()
        
    # ---------------- helpers ----------------
    def _soft_xent_loss(self, sim, soft_label):
        logprobs = F.log_softmax(sim, dim=-1)
        target = F.softmax(soft_label, dim=-1)
        return -(target * logprobs).sum()

    # ---------------- forward ----------------
    def forward(self, samples):
        feat1, feat2, feat3, feat4, feat5 = (samples["feat1"], samples["feat2"], samples["feat3"], samples["feat4"], samples["feat5"], )
        text = samples["abn_text_input"]
        abn_label = samples["abn_label"]
        device = feat1.device
        bs = abn_label.shape[0]
        abn_size = self.abn_size

        # ---- Classification ----
        pooled_feat = self.pool(feat5).squeeze()
        abn_logits = self.classifier(pooled_feat).reshape(bs, abn_size)
        abn_probs = torch.sigmoid(abn_logits)
        loss_cls = self.cls_loss_function(abn_logits, abn_label.float())

        # ---- Image Embedding ----
        feat1_embed = self.layer1(feat1).transpose(1, 2)
        feat2_embed = self.layer2(feat2).transpose(1, 2)
        feat3_embed = self.layer3(feat3).transpose(1, 2)
        feat4_embed = self.layer4(feat4).transpose(1, 2)
        feat5_embed = self.layer5(feat5).transpose(1, 2)
        cls_embed = self.cls_embedder(abn_probs[:, None])
        image_embeds = torch.cat([cls_embed, feat1_embed, feat2_embed, feat3_embed, feat4_embed, feat5_embed], dim=1,)
        f_dim = image_embeds.shape[1]
        image_embeds_a = image_embeds[:, None].expand(-1, abn_size, -1, -1).reshape(bs * abn_size, f_dim, 1408)# [img000,img111,img222,...]
        image_atts = torch.ones((bs * abn_size, f_dim), dtype=torch.long, device=device)

        # global abnormality learning queries
        query_tokens = self.query_tokens[None].expand(bs, -1, -1, -1).reshape(bs * abn_size, self.num_query_token, self.hidden_dim)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_a,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        # [b, num_query_token, hidden]
        image_abn_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
        image_feats = image_abn_feats.reshape(bs, abn_size, self.num_query_token, -1)

        # ---- Text Embedding ----
        text_tokens = self.tokenizer(
            list(chain(*text)),
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        if self.style_learn:
            # client text style learning queries
            # Style tokens: 每个client对应独立向量
            base_style = self.style_tokens[self.client_id].unsqueeze(0)  # [1, n_style_token, hidden]
            style_tokens = base_style.expand(bs * abn_size, self.n_style_token, self.hidden_dim)
            style_att_mask = torch.ones(bs * abn_size, self.n_style_token, device=device)
            text_att_mask = torch.cat([style_att_mask, text_tokens.attention_mask], dim=1)
        else:
            style_tokens = None
            text_att_mask = text_tokens.attention_mask

        text_output = self.Qformer.bert(
            input_ids=text_tokens.input_ids,
            style_query_embeds=style_tokens,
            attention_mask=text_att_mask,
            return_dict=True,
        )

        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, self.n_style_token + 0, :]), dim=-1) # [CLS] token
        text_feat = text_feat.reshape(bs, abn_size, -1)
        
        # ---- ITC loss ----
        # ============== Image-text Contrastive =================== #
        image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        image_feats_all = image_feats_all.permute(1,0,2,3)
        image_feats = image_feats.permute(1,0,2,3)
        text_feat_all = text_feat_all.permute(1, 0, 2)
        text_feat = text_feat.permute(1, 0, 2)
        # abnormality-wise image-text similarity in Matrix operations
        # [32, 30, 1, 10, 256] * [32, 1, 30, 256, 1] -> [32, 30, 30, 10, 1]
        sim_q2t = torch.matmul(
                image_feats.unsqueeze(2), text_feat_all.unsqueeze(1).unsqueeze(-1)
            ).squeeze() 
        sim_i2t= sim_q2t
        sim_i2t = sim_i2t / self.temp
        
        # # [32, 30, 1, 1, 256] * [32, 1, 30, 256, 10] -> [32, 30, 30, 10, 1]
        sim_t2q = torch.matmul(
                text_feat.unsqueeze(2).unsqueeze(2), image_feats_all.permute(0, 1, 3, 2).unsqueeze(1)
            ).squeeze() 
        sim_t2i = sim_t2q
        sim_t2i = sim_t2i / self.temp     

        # [32, 30, 1] * [32, 1, 30] -> [32, 30, 30, 1]
        abn_probs_all = concat_all_gather(abn_probs)
        abn_probs = abn_probs.permute(1, 0)
        abn_probs_all = abn_probs_all.permute(1, 0)
        sim_abn_prob = torch.matmul(abn_probs.unsqueeze(2), abn_probs_all.unsqueeze(1))

        loss_itc = (
            self._soft_xent_loss(sim_i2t, sim_abn_prob) / (bs * abn_size)
            + self._soft_xent_loss(sim_t2i, sim_abn_prob) / (bs * abn_size)
        ) / 2

        # ================= Image Captioning ======================== #
        abn_labels = abn_label.reshape(-1)
        abn_index = torch.where(abn_labels == 1)[0]
        normal_pool = torch.where(abn_labels == 0)[0]
        
        # The number of normal resample to be consistent with abnormal sentences.
        abn_num = max(len(abn_index), 1)
        n_index = normal_pool[torch.randint(0, len(normal_pool), (abn_num,), device=device)]
        cap_index = torch.cat([abn_index, n_index], dim=0)

        cap_choice = torch.randint(0, len(cap_index), (min(max(len(cap_index), 1), 200),)) # 300 -> 200
        cap_index = cap_index[cap_choice]
        
        decoder_input_ids = text_tokens.input_ids.clone()[cap_index]
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        # queried image feats
        query_atts = torch.ones(query_tokens[cap_index].size()[:-1], dtype=torch.long).to(device)
        past_key_values_list = []
        for v in query_output.past_key_values:
            past_key_values_list.append((v[0][cap_index], v[1][cap_index]))

        if self.style_learn:
            style_tokens_lm = style_tokens[cap_index]
            style_atts = torch.ones(style_tokens_lm.size()[:-1], dtype=torch.long).to(device)
            text_att_mask_lm = torch.cat([style_atts, text_tokens.attention_mask[cap_index]], dim=1)
        else:
            style_tokens_lm = None
            text_att_mask_lm = text_tokens.attention_mask[cap_index]

        attention_mask = torch.cat([query_atts, text_att_mask_lm], dim=1)
        lm_output = self.Qformer(
            input_ids=decoder_input_ids,
            style_query_embeds=style_tokens_lm,
            attention_mask=attention_mask,
            past_key_values=past_key_values_list,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss
        # logits_output = lm_output.logits  # [cap_B, max_txt_len, 30523]
        # sequence_output = lm_output.hidden_states  # [cap_B, max_txt_len, hidden_dim]
        
        # ================= debug =================  # 
        # text_bs = np.array(list(chain(*text)))
        # text_lm = text_bs[cap_index.cpu().numpy()]
        # 1. 取最大概率的 token id
        # from common.utils import decode_from_logits
        # pred_texts = decode_from_logits(self.tokenizer, lm_output.logits)  
        # pred_ids = lm_output.logits.argmax(dim=-1)   # [B, L]
        # for r,p in zip(text_lm, pred_texts):
        #     print(r, "****", p)

        
        # ================= Text Style Contrastive ================== #
        if self.style_learn:
            old = self.style_center[self.client_id].clone() # debug
            abn_labels = abn_label.reshape(-1)
            abn_index = torch.where(abn_labels==1)[0]
            
            if len(abn_index) == 0:
                loss_style = torch.tensor(0.0, device=device)
            else:
                # 1. 采样 abnormal 句子
                # tsc_num = 200
                # abn_choice = torch.randint(0, len(abn_index), (tsc_num,), device=device)
                # abn_index = abn_index[abn_choice]
                # update 12252025
                tsc_num = min(200, len(abn_index))
                perm = torch.randperm(len(abn_index), device=device)[:tsc_num]
                abn_index = abn_index[perm]
                
                # 2. 当前 client 的 style embedding（有梯度）
                # text_output 已经是用本 client 的 style_tokens 跑过 Qformer 的输出
                self_style  = self.text_style_proj(text_output.last_hidden_state[abn_index][:, 0:self.n_style_token, :])
                
                # 3. 本机多 GPU 聚合：batch-level style center for this client
                #    先对 (sample, token) 平均，得到一个 D 向量
                local_center = self_style.mean(dim=(0, 1))  # [D]
                local_center = local_center.unsqueeze(0)    # [1, D]

                # 5) 多卡 all-gather，同一个 client 的多个 rank 做全局平均(有梯度)
                local_center_all = all_gather_with_grad(local_center)     # [world_size,1,D] or [world_size,D]
                if local_center_all.dim() == 3:
                    local_center_all = local_center_all.squeeze(1)        # [W, D]
                cur_vec = F.normalize(local_center_all.mean(dim=0), dim=-1)      # [D]  

                # 6) EMA 更新 center（无梯度）
                with torch.no_grad():
                    if self.style_center[self.client_id].abs().sum() == 0:
                        # 第一次更新：直接用当前 batch 的向量初始化
                        self.style_center[self.client_id].copy_(cur_vec)
                    else:
                        m = self.style_momentum  # 来自 LOCAL_MOMENTUM
                        self.style_center[self.client_id].mul_(m).add_(cur_vec * (1.0 - m))

                # 7) 对齐损失：让当前 batch 的 style 向量接近自己的 EMA 中心
                # center_detached = self.style_center[self.client_id].detach()  # [D]
                # loss_align = F.mse_loss(cur_vec, center_detached)
                # update 12242025
                center_detached = F.normalize(self.style_center[self.client_id].detach(), dim=-1, eps=1e-6)
                loss_align = torch.relu(0.9 - (cur_vec * center_detached).sum())# cos>=0.95 不再约束
                
                # 6. Push：用全局 style_center 做 margin 分离
                # centers_detached = F.normalize(self.style_center.detach(), dim=-1)  # [num_clients, D]
                # other_ids = [i for i in range(self.num_clients) if i != self.client_id]
                # update 12252025 separation：只对已初始化的 other centers
                centers_detached = F.normalize(self.style_center.detach(), dim=-1, eps=1e-6)
                other_ids = [i for i in range(self.num_clients) if i != self.client_id]

                if len(other_ids) > 0:
                    other_centers = centers_detached[other_ids]            # [C-1, D]
                    # cos_sim = (centers[self.client_id].unsqueeze(0) * other_centers).sum(dim=-1)  # [C-1]
                    # 用“当前 batch style 向量”对其它 client 的 center 做 margin 对比
                    cos_sim = (cur_vec.unsqueeze(0) * other_centers).sum(dim=-1)  # [C-1]
                    dist = 1.0 - cos_sim
                    diff = torch.relu(STYLE_MARGIN - dist)
                    loss_sep = (diff * diff).mean()
                else:
                    loss_sep = torch.tensor(0.0, device=device)
                
                # ------- 6. 最终 style loss -------
                loss_style = (loss_sep + loss_align) * STYLE_LAMBDA

                
                #  debug
                with torch.no_grad():
                    new = self.style_center[self.client_id]
                    # ---- 1) 本 client：更新幅度 ----
                    delta_l2 = (new - old).norm(p=2)                 # []
                    delta_l2_mean = concat_all_gather(delta_l2.view(1)).mean()

                    old_n = F.normalize(old, dim=0, eps=1e-6)
                    new_n = F.normalize(new, dim=0, eps=1e-6)
                    cos = (old_n * new_n).sum()                      # []
                    cos_mean = concat_all_gather(cos.view(1)).mean()
                    
                    # ---- 2) 跨 client：center 之间相似度（是否塌缩）----
                    centers = self.style_center.detach()             # [C, D]
                    centers_n = F.normalize(centers, dim=-1, eps=1e-6)  # [C, D]

                    # 当前 client 与所有 centers 的 cos
                    sims = (centers_n @ new_n.view(-1, 1)).squeeze(1)   # [C]
                    # 去掉自己
                    mask = torch.ones_like(sims, dtype=torch.bool)
                    mask[self.client_id] = False
                    other_sims = sims[mask]                              # [C-1]

                    if other_sims.numel() > 0:
                        inter_cos_max = other_sims.max()                 # []
                        inter_cos_mean = other_sims.mean()               # []
                        inter_cos_min = other_sims.min()                 # []
                    else:
                        inter_cos_max = torch.tensor(0.0, device=new.device)
                        inter_cos_mean = torch.tensor(0.0, device=new.device)
                        inter_cos_min = torch.tensor(0.0, device=new.device)

                    # 也可以看“全体 centers”两两相似度（不只当前 client）
                    # sim_mat = centers_n @ centers_n.t()                # [C, C]
                    # sim_mat.fill_diagonal_(0)
                    # global_inter_max = sim_mat.max()                   # []
                    # global_inter_mean = sim_mat.sum() / (C*C - C)      # []

                    # ---- 3) 分布式聚合（取各 rank 均值）----
                    inter_cos_max_mean  = concat_all_gather(inter_cos_max.view(1)).mean()
                    inter_cos_mean_mean = concat_all_gather(inter_cos_mean.view(1)).mean()
                    inter_cos_min_mean  = concat_all_gather(inter_cos_min.view(1)).mean()                
        else:
            loss_style = torch.tensor(0.0, device=device)
            loss_sep = torch.tensor(0.0, device=device)
            loss_align = torch.tensor(0.0, device=device)
            cos_mean = torch.tensor(0.0, device=device)
            delta_l2_mean = torch.tensor(0.0, device=device)
            inter_cos_max_mean = torch.tensor(0.0, device=device)
            inter_cos_mean_mean = torch.tensor(0.0, device=device)
            inter_cos_min_mean = torch.tensor(0.0, device=device)
        
        loss_total = loss_itc + loss_cls + loss_lm
        
        return {
            "loss": loss_total,
            "loss_itc": loss_itc,
            "loss_cls": loss_cls,
            "loss_lm": loss_lm,
            "loss_style": loss_style,
            "loss_sep": loss_sep,
            "loss_align": loss_align,
            "center_cos": cos_mean,
            "center_delta_l2": delta_l2_mean,
            "inter_cos_max": inter_cos_max_mean,
            "inter_cos_mean": inter_cos_mean_mean,
            "inter_cos_min": inter_cos_min_mean,
        }

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - feat1..feat5: 3D feature tensors
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions_out: list, 每个样本对应 [captions(abn_size), probs(abn_size)]
        """

        feat1 = samples["feat1"]
        feat2 = samples["feat2"]
        feat3 = samples["feat3"]
        feat4 = samples["feat4"]
        feat5 = samples["feat5"]
        device = feat1.device
        bs = feat1.shape[0]
        abn_size = self.abn_size

        pooled_feat = self.pool(feat5).squeeze()
        abn_logits = self.classifier(pooled_feat).reshape(bs, abn_size)
        abn_logits = abn_logits.reshape(bs, self.abn_size)
        abn_probs = torch.sigmoid(abn_logits)

        feat1_embed = self.layer1(feat1).transpose(1, 2)
        feat2_embed = self.layer2(feat2).transpose(1, 2)
        feat3_embed = self.layer3(feat3).transpose(1, 2)
        feat4_embed = self.layer4(feat4).transpose(1, 2)
        feat5_embed = self.layer5(feat5).transpose(1, 2)
        cls_embed = self.cls_embedder(abn_probs[:, None])
        image_embeds = torch.concat(
            [cls_embed, feat1_embed, feat2_embed, feat3_embed, feat4_embed, feat5_embed],
            dim=1,
        )
        f_dim = image_embeds.shape[1]
        image_embeds_a = (
            image_embeds[:, None]
            .expand(-1, abn_size, -1, -1)
            .reshape(bs * abn_size, f_dim, 1408)
        )

        # global abnormality learning queries
        query_tokens = (
            self.query_tokens[None]
            .expand(bs, -1, -1, -1)
            .reshape(bs * abn_size, self.num_query_token, self.hidden_dim)
        )

        image_atts = torch.ones(image_embeds_a.size()[:-1], dtype=torch.long).to(device)

        model_kwargs = {
            "encoder_hidden_states": image_embeds_a,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(bs * abn_size, 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )

        if self.style_learn:
            base_style = self.style_tokens[self.client_id].unsqueeze(0)  # [1, n_style_token, hidden]
            style_tokens = base_style.expand(bs * abn_size, self.n_style_token, self.hidden_dim)
        else:
            style_tokens = None

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            style_query_embeds=style_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions_list = np.array(captions).reshape(bs, abn_size)

        captions_list = captions_list[:, None]
        abn_probs_list = abn_probs.detach().cpu().numpy().reshape(bs, 1, abn_size)

        captions_out = np.concatenate([captions_list, abn_probs_list], axis=1).tolist()
        return captions_out


