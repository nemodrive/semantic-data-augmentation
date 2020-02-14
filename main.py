from torchvision import datasets, transforms
import cityscapes
import matplotlib.pyplot as plt
import numpy as np
from cs_sematic_extractor import CSSemanticExtractor


def main():
    csse = CSSemanticExtractor('./resources/datasets/semanticsets', 10000)

    train_set = cityscapes.Cityscapes(
        root='./resources/datasets/cityscapes/',
        login=['none', 'none'],
        split='train',
        mode='fine',
        target_type=['semantic', 'polygon'],
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=True
    )

    stats = csse(train_set, all_instances=True, show_progress=True)
    print(stats)

    # valid_set = cityscapes.Cityscapes(
    #     root='./resources/datasets/cityscapes/',
    #     login=['none', 'none'],
    #     split='val',
    #     mode='fine',
    #     target_type=['semantic', 'polygon'],
    #     transform=transforms.Compose([
    #         transforms.ToTensor()
    #     ]),
    #     download=True
    # )
    #
    # stats = csse(valid_set, all_instances=False, show_progress=True)
    # print(stats)

    # TODO: remove this when done

    image, target = train_set[3]
    # count = 0
    # for d in target[1]['objects']:
    #     if d['label'] == 'person':
    #         count += 1
    # print(count)

    # fig = plt.figure(figsize=[20, 5])
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(image.permute(1, 2, 0))
    # # fig.add_subplot(1, 2, 2)
    # # plt.imshow(np.asanyarray(target[0]))
    # plt.tight_layout()
    # plt.show(block=True)

    return


if __name__ == '__main__':
    main()
