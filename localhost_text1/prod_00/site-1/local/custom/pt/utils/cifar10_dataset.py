# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torchvision import datasets
from pt.utils.cifar10_data import CustomCIFAR10Dataset


class CIFAR10_Idx(torch.utils.data.Dataset):
    def __init__(self, root, data_idx=None, transform=None):
        """CIFAR-10 dataset with index to extract subset

        Args:
            root: data root
            data_idx: to specify the data for a particular client site.
                If index provided, extract subset, otherwise use the whole set
            train: whether to use the training or validation split (default: True)
            transform: image transforms
            download: whether to download the data (default: False)
        Returns:
            A PyTorch dataset
        """
        self.root = root
        self.data_idx = data_idx
        self.transform = transform
        self.data, self.target = self.__build_cifar_subset__()

    def __build_cifar_subset__(self):
        # if index provided, extract subset, otherwise use the whole set
        cifar_dataobj = CustomCIFAR10Dataset(self.root, self.transform)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.labels)
        if self.data_idx is not None:
            if isinstance(self.data_idx, list):
                data = [data[i] for i in self.data_idx]
                target = target[self.data_idx]
            else:
                data = data[self.data_idx]
                target = target[self.data_idx]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
