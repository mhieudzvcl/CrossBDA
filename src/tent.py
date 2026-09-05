"""
tent.py - Test-Time Entropy Minimization (TENT)
Wang et al., ICLR 2021: https://arxiv.org/abs/2006.10726
"""
import torch
import torch.nn as nn
from copy import deepcopy


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np_, p in m.named_parameters():
                if np_ in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def softmax_entropy(logits):
    p = logits.softmax(dim=1)
    log_p = logits.log_softmax(dim=1)
    return -(p * log_p).sum(dim=1).mean()


class TENT(nn.Module):
    def __init__(self, model, lr=1e-3, steps=1):
        super().__init__()
        self.model = configure_model(model)
        params, _ = collect_params(self.model)
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
        self.steps = steps
        self._model_state = deepcopy(self.model.state_dict())
        self._optimizer_state = deepcopy(self.optimizer.state_dict())

    def forward(self, pre_img, post_img):
        for _ in range(self.steps):
            loc_logits, dmg_logits = self.model(pre_img, post_img)
            loss = softmax_entropy(loc_logits) + softmax_entropy(dmg_logits)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        with torch.no_grad():
            loc_logits, dmg_logits = self.model(pre_img, post_img)
        return loc_logits, dmg_logits

    def reset(self):
        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optimizer_state)
