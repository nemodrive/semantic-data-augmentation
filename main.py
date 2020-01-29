from torchvision import datasets, transforms
import cityscapes


def main():
    train_set = cityscapes.Cityscapes(
        root='./resources/datasets/cityscapes/',
        login=['amilab', 'amilab@aimas'],
        split='train',
        mode='fine',
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=False
    )

    return


if __name__ == '__main__':
    main()
