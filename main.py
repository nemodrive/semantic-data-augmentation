from torchvision import datasets, transforms
import cityscapes
import matplotlib.pyplot as plt
import numpy as np


def main():
    train_set = cityscapes.Cityscapes(
        root='./resources/datasets/cityscapes/',
        login=['amilab', 'amilab@aimas'],
        split='train',
        mode='fine',
        target_type=['semantic', 'polygon'],
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=True
    )

    img, (inst, poly) = train_set[3]

    count = 0
    for d in poly['objects']:
        if d['label'] == 'person':
            count += 1
    print(count)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    fig.add_subplot(1, 2, 2)
    plt.imshow(np.asanyarray(inst))
    plt.show(block=True)

    return


if __name__ == '__main__':
    main()
