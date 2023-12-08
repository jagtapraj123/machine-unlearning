import os
import numpy as np
import pandas as pd

from utils.preprocessing import Preprocessor

import torch
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR100
from torchvision import transforms


class LabeledDatasetMapper(Dataset):
    """
    Dataset Mapper class to get image and label given an index from Labeled Dataset.

    The class is written using torch parent class 'Dataset' for parallelizing and prefetching to accelerate training

    -----------
    Attributes:
    -----------
    - root_dir: str
        - directory path where images are stored

    - image_data_file: str
        - path to csv file that stores image names and labels

    - preprocessor: Instance of subclass of utils.Preprocessor
        - stores preprocessor with 'get' function that returns processed image given path to image and transformations

    - augment: Bool
        - boolean value specifying whether to transform images in preprocessor.
    """

    def __init__(
        self,
        root_dir: str,
        image_data_file: str,
        preprocessor: Preprocessor,
        augment: bool = False,
    ) -> None:
        """
        Init for LabeledDatasetMapper

        -----
        Args:
        -----
        - root_dir: str
            - directory path where images are stored

        - image_data_file: str
            - path to csv file that stores image names and labels

        - preprocessor: Instance of subclass of utils.Preprocessor
            - stores preprocessor with 'get' function that returns processed image given path to image and transformations

        - augment: Bool
            - boolean value specifying whether to transform images in preprocessor.
        """

        self.root_dir = root_dir
        self.image_data_file = pd.read_csv(image_data_file)
        self.num_classes = len(self.image_data_file["label"].unique())
        self.preprocessor = preprocessor
        self.augment = augment

    def __len__(self):
        """
        Function to get size of dataset
        """

        return self.image_data_file.shape[0]

    def __getitem__(self, idx):
        """
        Mapper function to get processed image and label given an index

        -----
        Args:
        -----
        - idx: int (python int / numpy.int / torch.int)
            - index of an image
            - idx >= 0 and idx < self.__len__()
        """

        # Convert to python int
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get final image path from image data csv file
        img_name = os.path.join(self.root_dir, self.image_data_file.iloc[idx, 0])

        # Get processed image from preprocessor given image path
        if self.augment:
            image = self.preprocessor.get(
                self.preprocessor.make_random_combinations(
                    1,
                    p_transformations={
                        "rotate": 0.5,
                        "scale": 0.5,
                        "flip": 0.5,
                        "gaussian_blur": 0.1,
                        "color_jitter": 0,
                        "random_erasing": 0,
                    },
                )[0],
                image_path=img_name,
                color_jitter=None,
                rotate=np.random.randint(0, 45),
                scale=np.random.uniform(0.7, 1),
                flip="h",
                gaussian_blur=1,
            )
        else:
            image = self.preprocessor.get("", img_name)

        img_label = self.image_data_file.iloc[idx, 1]

        return image, img_label

class LabeledPickleDatasetMapper(Dataset):
    """
    Dataset Mapper class to get image and label given an index from Labeled Dataset.

    The class is written using torch parent class 'Dataset' for parallelizing and prefetching to accelerate training

    -----------
    Attributes:
    -----------
    - root_dir: str
        - directory path where images are stored

    - image_data_file: str
        - path to csv file that stores image names and labels

    - preprocessor: Instance of subclass of utils.Preprocessor
        - stores preprocessor with 'get' function that returns processed image given path to image and transformations

    - augment: Bool
        - boolean value specifying whether to transform images in preprocessor.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        image_data_file: str or None,
        preprocessor: Preprocessor,
        augment: bool = False,
        return_idx: bool = False
    ) -> None:
        """
        Init for LabeledDatasetMapper

        -----
        Args:
        -----
        - root_dir: str
            - directory path where images are stored

        - image_data_file: str
            - path to csv file that stores image names and labels

        - preprocessor: Instance of subclass of utils.Preprocessor
            - stores preprocessor with 'get' function that returns processed image given path to image and transformations

        - augment: Bool
            - boolean value specifying whether to transform images in preprocessor.
        """

        # self.data = data
        self.data = np.zeros((data.shape[0], 3,32,32))
        for i in range(data.shape[0]):
            self.data[i] = data[i].reshape((3,32,32))
        
        self.labels = labels
        
        if image_data_file is None:
            self.subset_file = None
        else:
            self.subset_file = pd.read_csv(image_data_file)
        
        self.num_classes = 100
        self.preprocessor = preprocessor
        self.augment = augment
        self.return_idx = return_idx

    def __len__(self):
        """
        Function to get size of dataset
        """
        if self.subset_file is not None:
            return self.subset_file.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Mapper function to get processed image and label given an index

        -----
        Args:
        -----
        - idx: int (python int / numpy.int / torch.int)
            - index of an image
            - idx >= 0 and idx < self.__len__()
        """

        # Convert to python int
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get final image path from image data csv file
        # img_name = os.path.join(self.root_dir, self.image_data_file.iloc[idx, 0])
        
        if self.subset_file is not None:
            img_idx = self.subset_file.iloc[idx, 0]
        else:
            img_idx = idx

        # image = self.data[img_idx].reshape((3,32,32))
        image = self.data[img_idx]

        # Get processed image from preprocessor given image path
        if self.augment:
            image = self.preprocessor.get(
                self.preprocessor.make_random_combinations(
                    1,
                    p_transformations={
                        "rotate": 0,
                        "scale": 0.5,
                        "flip": 0.5,
                        "gaussian_blur": 0,
                        "color_jitter": 0,
                        "random_erasing": 0,
                    },
                )[0],
                # image_path=img_name,
                image_data=image,
                color_jitter=None,
                rotate=np.random.randint(0, 15),
                scale=np.random.uniform(0.85, 1),
                flip="h",
                gaussian_blur=0,
            )
        else:
            image = self.preprocessor.get("", image_data=image)

        # img_label = self.image_data_file.iloc[idx, 1]
        img_label = self.labels[img_idx]

        if self.return_idx:
            return image, img_label, img_idx
        return image, img_label


class GenDatasetMapper(Dataset):
    """
    Dataset Mapper class to get an image at random index and a dummy label.

    The class is written using torch parent class 'Dataset' for parallelizing and prefetching to accelerate training

    -----------
    Attributes:
    -----------
    - root_dir: str
        - directory path where images are stored

    - list_dir: list[str]
        - list of all the images present in the root_dir

    - set_size: int
        - number of all the images present in the root_dir

    - preprocessor: Instance of subclass of utils.Preprocessor
        - stores preprocessor with 'get' function that returns processed image given path to image and transformations

    - pretrain_size: int
        - number of images to use in one epoch
        - pretrain_size is returned when __len__ is called which makes the DataLoader only retrieve pretrain_size number of images.

    - augment: Bool
        - boolean value specifying whether to transform images in preprocessor.

    - last_img_given: int
        - index of the image sent to DataLoader in previous call.
        - this attribute is only used for book-keeping
    """

    def __init__(
        self,
        root_dir: str,
        preprocessor: Preprocessor,
        pretrain_size: int = 1,
        augment: bool = False,
    ) -> None:
        """
        Init for GenDatasetMapper

        -----
        Args:
        -----
        - root_dir: str
            - directory path where images are stored

        - preprocessor: Instance of subclass of utils.Preprocessor
            - stores preprocessor with 'get' function that returns processed image given path to image and transformations

        - pretrain_size: int
            - number of images to use in one epoch
            - pretrain_size is returned when __len__ is called which makes the DataLoader only retrieve pretrain_size number of images.

        - augment: Bool
            - boolean value specifying whether to transform images in preprocessor.
        """

        self.root_dir = root_dir
        self.list_dir = os.listdir(root_dir)
        self.set_size = len(self.list_dir)
        print("**\ngen_data size: {}\n**".format(self.set_size))
        self.preprocessor = preprocessor
        self.pretrain_size = pretrain_size
        self.augment = augment
        self.last_img_given = -1

    def __len__(self):
        """
        Function to get size of dataset for each epoch as specified (pretrain_size)
        """

        return self.pretrain_size

    def __getitem__(self, _):
        """
        Mapper function to get a processed image at random index and a dummy label
        """

        # print(idx, flush=True)

        self.last_img_given = np.random.randint(low=0, high=102400)

        # print(self.last_img_given, flush=True)
        # print(os.path.join(self.root_dir, self.list_dir[self.last_img_given]), flush=True)

        if self.augment:
            image = self.preprocessor.get(
                self.preprocessor.make_random_combinations(
                    1,
                    p_transformations={
                        "rotate": 0.5,
                        "scale": 0.5,
                        "flip": 0.5,
                        "gaussian_blur": 0.1,
                        "color_jitter": 0,
                        "random_erasing": 0,
                    },
                )[0],
                image_path=os.path.join(
                    self.root_dir, self.list_dir[self.last_img_given]
                ),
                color_jitter=None,
                rotate=np.random.randint(0, 45),
                scale=np.random.uniform(0.7, 1),
                flip="h",
                gaussian_blur=1,
            )
        else:
            image = self.preprocessor.get(
                "",
                image_path=os.path.join(
                    self.root_dir, self.list_dir[self.last_img_given]
                ),
            )

        img_features = 0

        return image, img_features


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class CustomCIFAR100(CIFAR100):
    coarse_map = {
            0:[4, 30, 55, 72, 95],
            1:[1, 32, 67, 73, 91],
            2:[54, 62, 70, 82, 92],
            3:[9, 10, 16, 28, 61],
            4:[0, 51, 53, 57, 83],
            5:[22, 39, 40, 86, 87],
            6:[5, 20, 25, 84, 94],
            7:[6, 7, 14, 18, 24],
            8:[3, 42, 43, 88, 97],
            9:[12, 17, 37, 68, 76],
            10:[23, 33, 49, 60, 71],
            11:[15, 19, 21, 31, 38],
            12:[34, 63, 64, 66, 75],
            13:[26, 45, 77, 79, 99],
            14:[2, 11, 35, 46, 98],
            15:[27, 29, 44, 78, 93],
            16:[36, 50, 65, 74, 80],
            17:[47, 52, 56, 59, 96],
            18:[8, 13, 48, 58, 90],
            19:[41, 69, 81, 85, 89]
        }
    def __init__(self, root, train, download, transform, coarse_labels=False):
        super().__init__(root = root, train = train, download = download, transform = transform)
        self.coarse_labels = coarse_labels
        # self.coarse_map = {
        #     0:[4, 30, 55, 72, 95],
        #     1:[1, 32, 67, 73, 91],
        #     2:[54, 62, 70, 82, 92],
        #     3:[9, 10, 16, 28, 61],
        #     4:[0, 51, 53, 57, 83],
        #     5:[22, 39, 40, 86, 87],
        #     6:[5, 20, 25, 84, 94],
        #     7:[6, 7, 14, 18, 24],
        #     8:[3, 42, 43, 88, 97],
        #     9:[12, 17, 37, 68, 76],
        #     10:[23, 33, 49, 60, 71],
        #     11:[15, 19, 21, 31, 38],
        #     12:[34, 63, 64, 66, 75],
        #     13:[26, 45, 77, 79, 99],
        #     14:[2, 11, 35, 46, 98],
        #     15:[27, 29, 44, 78, 93],
        #     16:[36, 50, 65, 74, 80],
        #     17:[47, 52, 56, 59, 96],
        #     18:[8, 13, 48, 58, 90],
        #     19:[41, 69, 81, 85, 89]
        # }
        
    #def __len__(self):
    #    len(self.main_dataset)
        
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if not self.coarse_labels:
            return x, y, index
        
        coarse_y = None
        for i in range(20):
            if y in self.coarse_map[i]:
                coarse_y = i
                break
        assert coarse_y != None, "coarse_y is None for index: {}, y: {}".format(index, y)

        return x, coarse_y, index
    
    
# class UnLearningData(Dataset):
#     def __init__(self, forget_data, retain_data):
#         super().__init__()
#         self.forget_data = forget_data
#         self.retain_data = retain_data
#         self.forget_len = len(forget_data)
#         self.retain_len = len(retain_data)

#     def __len__(self):
#         return self.retain_len + self.forget_len
    
#     def __getitem__(self, index):
#         if(index < self.forget_len):
#             x = self.forget_data[index][0]
#             y = 1
#             return x,y
#         else:
#             x = self.retain_data[index - self.forget_len][0]
#             y = 0
#             return x,y