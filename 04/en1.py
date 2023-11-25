import torch
from torch import nn

if __name__ == "__main__":

    # 1. (32 × 3 × 128 × 128) のテンソルを作成
    tensor = torch.ones((32, 3, 128, 128))
    print("=== problem 1 ===")
    print(repr(tensor.size()))

    # 2. 出力が (32 × 64 × 126 × 126) となるように畳み込みを定義し、1のテンソルに適用しろ
    conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    out2 = conv2(tensor)
    print("=== problem 2 ===")
    print(repr(out2.size()))

    # 3. 出力が (32 × 256 × 64 × 64) となるように畳み込みを定義し、1のテンソルに適用しろ
    conv3 = nn.Conv2d(in_channels=3, out_channels=256,kernel_size=3, stride=2, padding=1)
    out3 = conv3(tensor)
    print("=== problem 3 ===")
    print(repr(out3.size()))

    # 4. 2, 3でkernel_size = 5として同様の結果が得られるように畳み込みを定義しろ
    conv4_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=1)
    out4_1 = conv4_1(tensor)
    print("=== problem 4_1 ===")
    print(repr(out4_1.size()))
    conv4_2 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=2, padding=2)
    out4_2 = conv4_2(tensor)
    print("=== problem 4_2 ===")
    print(repr(out4_2.size()))
