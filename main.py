import torch
import torchvision
from torchvision import datasets, transforms
import cityscapes_downloader


def main():
    csd = cityscapes_downloader.CityscapesDownloader(
        root='./resources/datasets/cityscapes/',
        login={'user': 'amilab', 'pass': 'amilab@aimas'},
        packages=['gtFine', 'gtCoarse', 'leftImg8bit']
    )

    csd.download()

    # TODO: Remove this after testing
    train_set = datasets.Cityscapes(
        root='./resources/datasets/cityscapes/',
        split='train',
        mode='fine',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    return


if __name__ == '__main__':
    main()
