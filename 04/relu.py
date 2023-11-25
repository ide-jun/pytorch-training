import torch
from torch import nn

if __name__ == "__main__":

    # 入力用のテンソル定義
    _in = torch.tensor([
        [-3., -2., 5.],
        [16., 43., -1.],
        [18., 3.1, 56.]
    ]).unsqueeze(dim=0).unsqueeze(dim=0)

    # ReLU 定義 & 適用
    relu = nn.ReLU()
    print(repr(relu(_in)))