import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, dataset_dir):
        dir_path_resolved = Path(dataset_dir).resolve()
        dir_list = list(dir_path_resolved.glob("*"))
        self.img_list = list()
        for dir in dir_list:
            imgs_path = list(Path(dir).glob("*.png"))
            for img_path in imgs_path:
                self.img_list.append(img_path)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        return img
    
if __name__ == "__main__":
    my_dataset = MyDataset("./data")
    print("--- problem 1.1 ---")
    print(len(my_dataset))
    print("--- problem 1.2 ---")
    print(my_dataset[0].size)