from PIL import Image
from torchvision import transforms
import numpy as np

if __name__ == "__main__":
    img_path = "./data/dog.png"
    img = Image.open(img_path)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 変換を適用
    transformed_img = transform(img)
    print("before type:\n", type(img))
    # numpy配列に変換
    img_array = np.array(img)
    print("before size:\n", img_array.shape)
    
    print("after type:\n", type(transformed_img))
    print("after size:\n", transformed_img.size())
