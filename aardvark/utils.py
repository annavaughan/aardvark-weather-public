from collections import defaultdict
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def channels_to_2nd_dim(x):
    return x.permute(*([0, x.dim() - 1] + list(range(1, x.dim() - 1))))


def channels_to_final_dim(x):
    return x.permute(*([0] + list(range(2, x.dim())) + [1]))


def collate(tensor_list):
    out_dict = defaultdict()
    for k in tensor_list[0].keys():
        out_dict[k] = [t[k] for t in tensor_list]
        out_dict[k] = pad_sequence(
            out_dict[k],
            padding_value=np.nan,
            batch_first=True,
        )
    return out_dict
