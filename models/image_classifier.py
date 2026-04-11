import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'  
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet152, ResNet152_Weights

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


# ========= Region-wise pooling =========
@torch.no_grad()
def _ensure_region_scale(region_onehot, target_shape, device, dtype):
    """
    将 [B,R,H,W,D] 的 onehot 区域图（0/1）插值到 target_shape=(H,W,D)
    使用最近邻以保持 onehot 性质
    """
    if region_onehot.shape[2:] == target_shape:
        return region_onehot

    # 最近邻插值需要 float
    region_onehot_f = region_onehot.float()
    region_onehot_f = F.interpolate(
        region_onehot_f, size=target_shape, mode="nearest"
    )
    return (region_onehot_f > 0.5).to(device=device, dtype=dtype)


def regionwise_pooling_safe_onehot(skip_feats, regions_onehot_list):
    """
    多尺度 region-wise pooling
    skip_feats: list of [B,C,H,W,D]，各层特征
    regions_onehot_list: list of [B,R,H?,W?,D?]，各层对应的 onehot 区域图（R=region_nums）
    返回: list of [B,R,C]，各层区域平均池化后的特征
    """
    region_feats_multi_scale = []
    assert len(skip_feats) >= 1, "skip_feats 为空"
    assert len(regions_onehot_list) >= 1, "regions_onehot_list 为空"

    for i_f, f in enumerate(skip_feats):
        B, C, H, W, D = f.shape
        device = f.device
        # 匹配一个regions_onehot（允许长度不等，超出取最后一个）
        region_onehot = regions_onehot_list[min(i_f, len(regions_onehot_list)-1)]
        # 期望 [B,R,H,W,D]
        assert region_onehot.dim() == 5, f"region_onehot dim 期望 5, got {region_onehot.shape}"

        # 插值到当前特征尺度
        region_onehot = _ensure_region_scale(region_onehot, (H, W, D), device=device, dtype=torch.bool)

        # 统计每个区域像素数: [B,R,1]
        counts = region_onehot.flatten(2).sum(-1, keepdim=True).clamp_min(1)

        # 将 f 正则化防止爆梯（不改变方向）：[B,C,H,W,D]
        f_norm = f / (f.abs().amax(dim=(2,3,4), keepdim=True) + 1e-6)

        # 扁平化：region_onehot [B,R,HWd]，f_flat [B,HWd,C]
        region_onehot_f = region_onehot.flatten(2).to(dtype=f_norm.dtype)
        region_norm = region_onehot_f / counts.to(dtype=f_norm.dtype)
        f_flat = f_norm.flatten(2).transpose(1, 2)  # [B, HWd, C]

        # 区域平均池化： [B,R,HWd] @ [B,HWd,C] -> [B,R,C]
        # 放在 autocast 外也安全；这里尊重外层 autocast
        region_feat = torch.bmm(region_norm, f_flat)

        region_feats_multi_scale.append(region_feat)

    return region_feats_multi_scale


# ========= Global–Region Attention =========
class GlobalRegionAttention(nn.Module):
    """
    全局–器官交互注意力
    - organ_nums = 多个器官头（作为 multi-head 数）
    - region_nums = 每个 head 里的“区域维度”（注意力最后一维 softmax）
    - embed_dim = region_nums * organ_nums（需能被 organ_nums 整除）
    输入:
        pooled_feat: [B, Cg]
        region_feats: [B, region_nums, Cr] (Cr 是级联后的区域特征维度)
    输出:
        logits: [B, organ_nums, region_nums]
        attn_w: MHA 的注意力权重
    """
    def __init__(self, region_dim, global_dim=2048, region_nums=70, organ_nums=7):
        super().__init__()
        self.region_nums = region_nums
        self.organ_nums = organ_nums
        embed_dim = region_nums * organ_nums
        assert embed_dim % organ_nums == 0, "embed_dim 必须能被 organ_nums 整除"
        
        self.q = nn.Linear(global_dim, embed_dim, bias=True)
        self.k = nn.Linear(region_dim, embed_dim, bias=True)
        self.v = nn.Linear(region_dim, embed_dim, bias=True)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=organ_nums, batch_first=True)

    def forward(self, pooled_feat, region_feats):
        # pooled_feat: [B, Cg] -> [B,1,E]
        q = self.q(pooled_feat.unsqueeze(1))
        # region_feats: [B,R,Cr] -> [B,R,E]
        k = self.k(region_feats)
        v = self.v(region_feats)
        # 注意：MHA 的序列维度是中间维（这里 1 和 R）
        out, attn_w = self.attn(q, k, v)  # out:[B,1,E]
        B, _, E = out.shape
        # 还原为 [B, organ_nums, region_nums]
        out = out.view(B, 1, self.organ_nums, self.region_nums).squeeze(1)
        return out, attn_w


# ========= 主干模型（I3ResNet 外挂） =========
from models.i3res import I3ResNet

class ImageClassifier_BASE(nn.Module):
    def __init__(self, out_channels, region_nums=70, embed_dim=256, extract_feat=False):
        """
        out_channels: list[int]，每个器官的类别数
        region_nums: 区域数量（默认 70）
        embed_dim: 每层嵌入维度
        """
        super().__init__()
        self.extract_feat = extract_feat
        # 使用官方权重枚举，避免版本差异
        base = resnet152(weights=ResNet152_Weights.DEFAULT)

        self.i3_resnet = I3ResNet(
            copy.deepcopy(base), conv_class=False, return_skips=True, return_pool=True
        )
        del base

        global_dim = 2048
        self.num_organs = len(out_channels)
        self.abn_classifier_heads = MLPHead(global_dim, embed_dim, num_classes=sum(out_channels))

        # 额外：器官存在性/选择器（可选）
        # self.organ_classifier_heads = MLPHead(global_dim, embed_dim, num_classes=self.num_organs)

    def forward(self, image, _unused_region_mask, region_onehot_list):
        """
        image: [B,1,D,H,W] 或 [B,1,H,W,D]（以 I3ResNet 约定为准）
        region_onehot_list: list of [B,R,H?,W?,D?]  (R=region_nums)
        """
        skip_feats, pooled_feat = self.i3_resnet(image)  # skip_feats: list of [B,C,H,W,D]
        if self.extract_feat:
            return skip_feats, pooled_feat
        pooled_feat = pooled_feat.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 2048]

        x_cls_heads = self.abn_classifier_heads(pooled_feat)           # [B, num_organs]
        # organ_cls = self.organ_classifier_heads(pooled_feat)           # [B, num_organs]
        organ_cls = None
        return x_cls_heads, organ_cls


# ========= Quick self-test =========
if __name__ == '__main__':
    import sys
    sys.path.append('/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/')
    from abnormality_list import abnormality_dict

    region_classes = [len(v) for v in abnormality_dict.values()]

    # 造数据（注意 randint 上限为 2，包含 0/1）
    B = 2
    D, H, W = 160, 224, 224
    image_pt = torch.randn((B, 1, D, H, W), device='cuda')  # 根据 I3ResNet 的输入约定调整维度
    organ_mask = torch.zeros((B, 1, D, H, W), device='cuda')  # 未使用，占位

    # 生成与 skip 尺度大致对应的 4 个尺度（可根据 I3ResNet 返回的中间层实际尺寸修改）
    region_onehot_list = [
        torch.randint(0, 2, (B, 70, H//2,  W//2,  D), device='cuda', dtype=torch.long).bool(),
        torch.randint(0, 2, (B, 70, H//4,  W//4,  D//2), device='cuda', dtype=torch.long).bool(),
        torch.randint(0, 2, (B, 70, H//8,  W//8,  D//4), device='cuda', dtype=torch.long).bool(),
        torch.randint(0, 2, (B, 70, H//16, W//16, D//8), device='cuda', dtype=torch.long).bool(),
    ]

    model = ImageClassifier_BASE(region_classes, region_nums=70, embed_dim=256).cuda()
    model.eval()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        abn_logits, organ_logits = model(image_pt, organ_mask, region_onehot_list)
    print(abn_logits.shape, organ_logits.shape)
