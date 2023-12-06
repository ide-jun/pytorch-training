from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_path = "./data/dog.png"
    img = Image.open(img_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transformed_img = transform(img)
    
    plt.imshow(transformed_img.permute(1, 2, 0))
    plt.show()
    
    