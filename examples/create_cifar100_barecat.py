"""Create barecat archives for CIFAR-100.

Usage:
    python examples/create_cifar100_barecat.py [--data-dir /var/tmp/cifar100_data]

Creates two barecat archives:
  - cifar100.barecat: annotations (impath + label per example, metadata)
  - cifar100_images.barecat: JPEG-encoded images
"""

import argparse

import numpy as np
import torchvision
from barecat import Barecat, DecodedView


def create_cifar100_barecat(data_dir='/var/tmp/cifar100_data'):
    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)

    anno_path = f'{data_dir}/cifar100.barecat'
    img_path = f'{data_dir}/cifar100_images.barecat'

    with Barecat(anno_path, readonly=False, overwrite=True) as bc_anno, \
         Barecat(img_path, readonly=False, overwrite=True) as bc_img:
        dec_anno = DecodedView(bc_anno)
        dec_img = DecodedView(bc_img)

        dec_anno['metadata.msgpack'] = dict(
            class_names=train_ds.classes,
            num_classes=100,
        )

        idx = 0
        for split_name, dataset in [('train', train_ds), ('test', test_ds)]:
            for image, label in dataset:
                impath = f'{idx:05d}.jpg'
                dec_img[impath] = np.array(image)
                dec_anno[f'{split_name}/{impath}_{0:02d}.msgpack'] = dict(
                    impath=impath, label=int(label),
                )
                idx += 1

    print(f'Created {anno_path} ({len(train_ds) + len(test_ds)} annotations)')
    print(f'Created {img_path} ({idx} images)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create CIFAR-100 barecat archives')
    parser.add_argument('--data-dir', default='/var/tmp/cifar100_data')
    args = parser.parse_args()
    create_cifar100_barecat(args.data_dir)
