import torch
from model import MyModel

if __name__ == "__main__":

    # 入力のテンソルを定義
    in_tensor = torch.ones((32, 3, 128, 128))

    # モデルインスタンス作成
    model = MyModel()

    # 実行 & 結果確認
    out = model(in_tensor)
    print(repr(out.size()))