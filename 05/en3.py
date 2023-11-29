import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MyDataset(Dataset):
    
    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        dir_list = list(dir_path_resolved.glob("*"))
        self.img_list = list()
        for dir in dir_list:
            imgs_path = list(Path(dir).glob("*.png"))
            for img_path in imgs_path:
                self.img_list.append(img_path)
        self.transform = transforms.Compose({
            transforms.ToTensor(),
        })
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        
        # ファイル名からラベルを取得
        img_path = Path(img_path)
        parts = img_path.parts
        label = int(parts[-2])
        return img_tensor, label
    
if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    tensor, label = my_dataset[0]
    print("--- problem 1.1 ---")
    print(tensor.size())
    print("--- problem 1.2 ---")
    print(label)