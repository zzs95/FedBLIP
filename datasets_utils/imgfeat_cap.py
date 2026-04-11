"""
Optimized dataset for Abn-CLIP training and evaluation.
Compatible with mini-LAVIS simplified training framework.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict
from datasets_utils.normal_text_template import normal_text_list
from datasets_utils.abnormality_list_56 import abnormality_list


# ==============================================================
# Base Dataset (simplified, not requiring lavis.datasets.BaseDataset)
# ==============================================================
class ImgFeatCapDataset(Dataset):
    """
    Main training dataset for Abn-CLIP pretraining.
    Loads multi-scale pooled features, abnormality labels, and captions.
    """

    def __init__(self, vis_root, ann_path, vis_processor=None, text_processor=None):
        """
        Args:
            vis_root (str): Path to npy feature root.
            ann_path (str): Path to JSON annotations.
            vis_processor (callable): Optional visual transform.
            text_processor (callable): Optional text preprocessing function.
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor or (lambda x: x)

        # load annotation json
        with open(ann_path, "r") as f:
            self.annotation = [json.loads(line) for line in f] 
            # self.annotation = self.annotation[:300] # debug 
            
        # print(len(pd.DataFrame.from_dict(self.annotation)['AccessionNumber_md5'].unique()))
        # build accession to ID mapping
        self.img_ids = {}
        for idx, ann in enumerate(self.annotation):
            img_id = ann.get("AccessionNumber_md5", str(idx))
            self.img_ids[img_id] = idx

        self.normal_text_list = normal_text_list

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # ----- load npz features -----
        feat_path = os.path.join(self.vis_root, ann["image_feat"][1:])
        # feat_path = os.path.join(self.vis_root, ann["image_feat"]) # mix
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")

        try:
            image_feat = np.load(feat_path)
        except Exception as e:
            print(f"[Warning] Skipping {feat_path}: {e}")
            return None

        abn_label = np.array(ann["abn_label"]).astype(np.float32)

        # ----- prepare caption list -----
        caption_list = np.random.choice(self.normal_text_list, len(abnormality_list)).tolist()
        for i, abn_name in enumerate(abnormality_list):
            caption_list[i] = caption_list[i].replace("<ABN_FIND>", abn_name)

        # replace positive findings with true abnormality text
        for i, abn_idx in enumerate(np.argwhere(abn_label > 0.5).reshape(-1)):
            abn_text = ann["abn_text"][i].replace('\u202f', '')
            caption_list[abn_idx] = self.text_processor(abn_text)

        # ----- construct tensor sample -----
        return {
            "feat1": torch.from_numpy(image_feat["pooled_scale_1"].astype(np.float32)),
            "feat2": torch.from_numpy(image_feat["pooled_scale_2"].astype(np.float32)),
            "feat3": torch.from_numpy(image_feat["pooled_scale_3"].astype(np.float32)),
            "feat4": torch.from_numpy(image_feat["pooled_scale_4"].astype(np.float32)),
            "feat5": torch.from_numpy(image_feat["scale_5"].astype(np.float32)),
            "abn_text_input": caption_list,
            "abn_label": torch.from_numpy(abn_label),
            "image_id": self.img_ids.get(ann.get("AccessionNumber_md5", ""), index),
        }


# ==============================================================
# Evaluation dataset (for captioning / validation)
# ==============================================================
class ImgFeatCapEvalDataset(Dataset):
    def __init__(self, vis_root, ann_path, text_processor=None):
        self.vis_root = vis_root
        self.text_processor = text_processor or (lambda x: x)

        with open(ann_path, "r") as f:
            self.annotation = json.load(f)

        self.img_ids = {ann["AccessionNumber_md5"]: i for i, ann in enumerate(self.annotation)}

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        feat_path = os.path.join(self.vis_root, ann["image_feat"][1:])
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature: {feat_path}")
        image_feat = np.load(feat_path)

        abn_label = np.array(ann["abn_label"]).astype(np.float32)
        caption_list = [f"no findings of {a}." for a in abnormality_list]

        # override positive findings
        for i, abn_idx in enumerate(np.argwhere(abn_label > 0.5).reshape(-1)):
            caption_list[abn_idx] = self.text_processor(ann["abn_text"][i])

        return {
            "feat1": torch.from_numpy(image_feat["pooled_scale_1"].astype(np.float32)),
            "feat2": torch.from_numpy(image_feat["pooled_scale_2"].astype(np.float32)),
            "feat3": torch.from_numpy(image_feat["pooled_scale_3"].astype(np.float32)),
            "feat4": torch.from_numpy(image_feat["pooled_scale_4"].astype(np.float32)),
            "feat5": torch.from_numpy(image_feat["scale_5"].astype(np.float32)),
            "abn_text_input": caption_list,
            "abn_label": torch.from_numpy(abn_label),
            "image_id": ann["AccessionNumber_md5"],
            "image_path": ann["image_feat"],
        }


# ==============================================================
# Quick Test Section
# ==============================================================
if __name__ == "__main__":
    import tempfile
    import json

    # --- fake test data ---
    tmpdir = tempfile.mkdtemp()
    vis_root = tmpdir
    ann_path = os.path.join(tmpdir, "ann.json")

    # fake feature npz
    dummy_feat = {
        "pooled_scale_1": np.random.randn(1, 64, 16, 16, 16).astype(np.float32),
        "pooled_scale_2": np.random.randn(1, 256, 8, 8, 8).astype(np.float32),
        "pooled_scale_3": np.random.randn(1, 512, 4, 4, 4).astype(np.float32),
        "pooled_scale_4": np.random.randn(1, 1024, 2, 2, 2).astype(np.float32),
        "scale_5": np.random.randn(1, 2048, 7, 7, 10).astype(np.float32),
    }
    np.savez(os.path.join(tmpdir, "sample_feat.npz"), **dummy_feat)

    ann = [
        {
            "image_feat": "/sample_feat.npz",
            "AccessionNumber_md5": "case_0001",
            "abn_label": [1 if i % 5 == 0 else 0 for i in range(len(abnormality_list))],
            "abn_text": [f"finding of {a}" for a in abnormality_list],
        }
    ]
    with open(ann_path, "w") as f:
        json.dump(ann, f)

    # --- run dataset ---
    ds = ImgFeatCapDataset(vis_root=vis_root, ann_path=ann_path)
    sample = ds[0]

    print("=== Dataset Test Output ===")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: tensor {tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"{k}: list of len {len(v)}")
        else:
            print(f"{k}: {v}")
