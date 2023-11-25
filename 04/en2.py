import torch
from torch import nn

if __name__ == "__main__":

    # 1. 入力用のテンソルとして、(32 × 1024) のテンソルを定義しろ
    _in = torch.ones((32, 1024))
    print("=== problem 1 ===")
    print(repr(_in.size()))

    # 2. 出力が (32 × 256) となるように全結合層を定義しろ
    fc2 = nn.Linear(in_features=1024, out_features=256, bias=True)
    print("=== problem 2 ===")
    print(repr(fc2(_in).size()))

    # 3. 出力が (32 × 2048) となるように全結合層を定義しろ
    fc3 = nn.Linear(in_features=1024, out_features=2048, bias=True)
    print("=== problem 3 ===")
    print(repr(fc3(_in).size()))

    # おまけ
    # 2で作成されたテンソルを (32 × 16 × 16) の形状のテンソルに直せ
    print(" === extra ===")
    reshaped = torch.reshape(fc2(_in), (32, 16, 16))
    print(repr(reshaped.size()))