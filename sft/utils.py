import torch
import torch.nn as nn
import yaml


def load_yaml_cfg(cfg: str):
    with open(cfg, 'r') as file:
        config = yaml.safe_load(file)
    return config

def send_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: send_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [send_to_device(item, device) for item in data]
    else:
        return data

def length2attention_mask(lengths, max_seq_len=None):
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths, dtype=torch.int)

    assert lengths.ndim == 1
    max_seq_len = lengths.max().item() if max_seq_len is None else max_seq_len
    attention_mask = torch.arange(max_seq_len, device=lengths.device) < lengths.unsqueeze(-1)
    return attention_mask.int()


class StopLossOne:
    def __init__(self):
        self.lf_pos = nn.CrossEntropyLoss(ignore_index=-100)
        self.lf_neg = nn.CrossEntropyLoss(ignore_index=-100)

    def __call__(self, pred, tag):
        tag_pos = torch.full_like(tag, fill_value=-100)
        tag_neg = tag_pos.clone()
        tag_pos[tag==1] = 1
        scores = pred.detach().softmax(1)
        sco_neg = scores[:, 0].clone()
        sco_neg[tag!=0] = 999
        sco_idx = sco_neg.argmin(-1)
        tag_neg[torch.arange(tag_neg.shape[0]), sco_idx] = 0
        stop_loss = 0.5*self.lf_pos(pred, tag_pos) + 0.5*self.lf_neg(pred, tag_neg)

        likely_pos = scores[:, 1][tag_pos==1].mean()
        likely_neg = scores[:, 0][tag_neg == 0].mean()
        return stop_loss, {'likely_pos': likely_pos.item(), 'likely_neg': likely_neg.item()}


def set_module_train_state(module:nn.Module, freeze=True):
    for _, p in module.named_parameters():
        p.requires_grad = not freeze