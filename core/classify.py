"""
E2V Classification Head and Utilities

Functions copied verbatim from:
    - evaluate_comprehensive_2x32.py: E2VClassificationHead, load_e2v_head (lines 93-113)
    - analyze_sample_level.py: classify_with_details (lines 53-77)
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    E2V_HEAD_PATH, E2V_LABEL_TO_IDX, CODEBOOK_BIASED_MAP, CLASSIFIER_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================
# E2V Classification Head  (from evaluate_comprehensive_2x32.py L93-113)
# ============================================================

class E2VClassificationHead(nn.Module):
    """emotion2vec classification head (linear projection)."""
    def __init__(self, weight, bias):
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0])
        self.proj.weight.data = weight
        self.proj.bias.data = bias

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.proj(x)


class CustomClassifierAsE2V(nn.Module):
    """Wraps a dataset-specific LinearProbe to output E2V-compatible 9-class logits.

    Non-existent classes get -1e9 logits so they never win argmax.
    This makes the wrapper drop-in compatible with E2VClassificationHead.
    """

    def __init__(self, classifier_path: str, source: str, device: str = 'cuda'):
        super().__init__()
        ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
        emotions = ckpt['emotions']
        feature_dim = ckpt['feature_dim']
        num_classes = ckpt['num_classes']

        self.proj = nn.Linear(feature_dim, num_classes)
        self.proj.weight.data = ckpt['model_state_dict']['classifier.weight']
        self.proj.bias.data = ckpt['model_state_dict']['classifier.bias']

        biased_map = CODEBOOK_BIASED_MAP[source]
        self._idx_map = []
        for i, emo in enumerate(emotions):
            e2v_label = biased_map.get(emo)
            if e2v_label:
                self._idx_map.append((i, E2V_LABEL_TO_IDX[e2v_label]))

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        raw = self.proj(x)
        out = torch.full((raw.size(0), 9), -1e9, device=raw.device, dtype=raw.dtype)
        for cls_idx, e2v_idx in self._idx_map:
            out[:, e2v_idx] = raw[:, cls_idx]
        return out


def load_custom_head(source: str, ssl_model: str = 'e2v',
                     device: str = 'cuda') -> CustomClassifierAsE2V:
    """Load a dataset-specific classifier wrapped as E2V-compatible head."""
    path = CLASSIFIER_DIR / ssl_model / source / 'best_model.pt'
    head = CustomClassifierAsE2V(str(path), source, device)
    head = head.to(device).eval()
    logger.info(f"  Loaded custom classifier: {path.name} "
                f"({len(head._idx_map)} classes mapped to E2V)")
    return head


def load_e2v_head(model_path: str = None, device: str = 'cuda') -> E2VClassificationHead:
    """Load the pretrained E2V classification head.

    Args:
        model_path: Path to model.pt checkpoint. Defaults to E2V_HEAD_PATH.
        device: 'cuda' or 'cpu'
    """
    if model_path is None:
        model_path = E2V_HEAD_PATH
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_state = ckpt['model']
    weight = model_state['proj.weight']
    bias = model_state['proj.bias']
    head = E2VClassificationHead(weight, bias)
    head = head.to(device).eval()
    return head


# ============================================================
# Classification utilities  (from analyze_sample_level.py L53-77)
# ============================================================

def classify_with_details(
    feats: torch.Tensor,
    e2v_head: E2VClassificationHead,
    valid_labels: List[str],
    valid_indices: List[int],
    device: str = 'cuda',
) -> dict:
    """Classify and return full details: prediction, confidence, softmax distribution.

    Args:
        feats:         Input features [T, D] or [1, T, D]
        e2v_head:      E2V classification head
        valid_labels:  List of valid emotion labels (e.g. FAIR_EMOTIONS)
        valid_indices: Corresponding indices in E2V 9-class logits (e.g. FAIR_E2V_INDICES)
        device:        'cuda' or 'cpu'

    Returns:
        dict with keys: prediction, confidence, entropy, softmax
    """
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    feats = feats.to(device)
    with torch.no_grad():
        logits = e2v_head(feats)
        valid_logits = logits[:, valid_indices]
        probs = F.softmax(valid_logits, dim=-1).squeeze(0)
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        softmax_dist = {label: probs[i].item() for i, label in enumerate(valid_labels)}
    return {
        'prediction': valid_labels[pred_idx],
        'confidence': confidence,
        'entropy': entropy,
        'softmax': softmax_dist,
    }


def classify_simple(
    feats: torch.Tensor,
    e2v_head: E2VClassificationHead,
    valid_labels: List[str],
    valid_indices: List[int],
    device: str = 'cuda',
) -> str:
    """Quick classification -- returns only the predicted label.

    Copied from evaluate_comprehensive_2x32.py L270-280 (inline `classify` function).
    """
    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    feats = feats.to(device)
    with torch.no_grad():
        logits = e2v_head(feats)
        valid_logits = logits[:, valid_indices]
        probs = F.softmax(valid_logits, dim=-1).squeeze(0)
        pred_idx = probs.argmax().item()
        return valid_labels[pred_idx]
