from torchvision import transforms, datasets

def cifer_datasets():
    # データセットの読み込み
    train_data = datasets.CIFAR10(
        root="../",
        train=True,
        transform=transforms.ToTensor(),
        download=True)
    
    test_data  = datasets.CIFAR10(
        root="../",
        train=False,
        transform = transforms.ToTensor(),
        download=True)
    
    return train_data, test_data