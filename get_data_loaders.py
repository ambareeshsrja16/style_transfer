"""
To DO : Add custom data loader capabilities, As of now written to work on the local cluster

Obtain dataloaders for classic datasets by providing name or custom dataloaders by providing specifications



pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install Cython


"""

import socket, getpass
import torch
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader


def get_classic_dataset(name = 'COCO', from_cluster=True, image_size=(256,256), batch_size=4):
    """Downloads the named dataset from torchvision.datasets to current directory in this hierarchy:
        ./data/

    https://pytorch.org/docs/stable/torchvision/datasets.html#coco

    train_loader will iterate as image, num_captions
    """

    req_transform = transforms.Compose([transforms.CenterCrop(image_size),transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {'num_workers': 4, 'pin_memory': True}

    if name == 'COCO' and from_cluster:

        if socket.gethostname().startswith("asreekum") and getpass.getuser() == 'asreekum':
            dataset_root_dir = '../../../../../../../../../datasets/COCO-2017'
        else:
            dataset_root_dir = '/datasets/COCO-2017'

        train_set = tv.datasets.ImageFolder(root=dataset_root_dir, transform=req_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, **kwargs)

        return train_loader


if __name__ == '__main__':

    test_train_loader = get_classic_dataset()

    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for i,caption in test_train_loader:
        logging.debug(f"{i.shape}")
        break

