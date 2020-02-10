from torchvision import datasets, transforms
import cityscapes
import matplotlib.pyplot as plt
import numpy as np
from cs_sematic_extractor import CSSemanticExtractor


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

    # image, target = train_set[3]
    # count = 0
    # for d in target[1]['objects']:
    #     if d['label'] == 'person':
    #         count += 1
    # print(count)
    csse = CSSemanticExtractor([[1, 1], [1,1]], 'person')

    total_count = 0
    for image, target in train_set:
        count = csse.get_target_count(target)
        print('Count: ', count)
        total_count += count
    print(total_count)

    # fig = plt.figure(figsize=[20, 5])
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(image.permute(1, 2, 0))
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(np.asanyarray(target[0]))
    # plt.tight_layout()
    # plt.show(block=True)

    return


if __name__ == '__main__':
    main()
