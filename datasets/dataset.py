from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of images and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and returns both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 dataset_name,  # 데이터셋 이름을 인자로 추가
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 use_lb_crop=False,
                 use_ulb_crop=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args:
            data: x_data
            dataset_name: 데이터셋 이름 (예: "CIFAR10", "CIFAR100", "STL10")
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: 기본 데이터 변환 (global weak augmentation)
            use_ulb_crop: If True, weak/strong augmentation 이후 crop 추가
            strong_transform: strong augmentation 적용을 위한 변환
            crop_transform: global crop 적용 transform
            local_crop_transform: local crop 적용 transform
            onehot: If True, label을 one-hot vector로 변환
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.dataset_name = dataset_name.upper()  # 대문자로 변환하여 통일
        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.use_lb_crop = use_lb_crop
        self.use_ulb_crop = use_ulb_crop
        self.onehot = onehot
        
        # 데이터셋에 따른 crop 크기 설정
        if self.dataset_name in ["CIFAR10", "CIFAR100"]:
            self.global_crop_size = 32
            self.local_crop_size = 24  # CIFAR에서는 일반적으로 24x24 local crop 사용
        elif self.dataset_name == "STL10":
            self.global_crop_size = 96
            self.local_crop_size = 64  # STL10에서는 64x64 local crop 사용
        else:
            self.global_crop_size = 224  # 기본값 (ImageNet 같은 큰 이미지용)
            self.local_crop_size = 128  
            
        # Crop Transform 정의
        self.crop_transform = transforms.RandomResizedCrop(self.global_crop_size, scale=(0.5, 1.0))
        self.local_crop_transform = transforms.RandomResizedCrop(self.local_crop_size, scale=(0.2, 0.5))

        self.transform = transform
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

        self.to_pil = transforms.ToPILImage()  # Tensor를 PIL로 변환하는 Transform 추가

    def __getitem__(self, idx):
        """
        If strong augmentation is not used:
            return weak_augment_image, target
        Else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            img_w = self.transform(img)
            if not self.is_ulb 
                if self.use_lb_crop:
                    img_w = self.to_pil(img_w) if isinstance(img_w, torch.Tensor) else img_w
                    img_w_global_crop = self.crop_transform(img_w)
                    img_w_local_crop1 = self.local_crop_transform(img_w)
                    img_w_local_crop2 = self.local_crop_transform(img_w)

                    # Labeled Data는 Global/Local Crop을 `torch.cat()`으로 묶어줌
                    img_w = torch.cat([transforms.ToTensor()(img_w_global_crop),
                                    transforms.ToTensor()(img_w_local_crop1),
                                    transforms.ToTensor()(img_w_local_crop2)], dim=0)

                    # Index, Target도 맞춰야 함
                    target = torch.tensor([target, target, target]) if target is not None else None
                    idx = torch.tensor([idx, idx, idx])
                return idx, img_w, target
            else:
                if self.alg == 'fixmatch':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'flexmatch':
                    if self.use_ulb_crop:  #  self.use_ulb_crop으로 수정
                        img_s = self.strong_transform(img)

                        if isinstance(img_w, torch.Tensor):
                            img_w = self.to_pil(img_w)
                        if isinstance(img_s, torch.Tensor):
                            img_s = self.to_pil(img_s)

                        w_global_crop = self.crop_transform(img_w)
                        w_local_crop1 = self.local_crop_transform(img_w)
                        w_local_crop2 = self.local_crop_transform(img_w)

                        s_global_crop = self.crop_transform(img_s)
                        s_local_crop1 = self.local_crop_transform(img_s)
                        s_local_crop2 = self.local_crop_transform(img_s)

                        # Crop된 이미지들을 `torch.cat()`으로 묶음
                        img_w = torch.cat([transforms.ToTensor()(w_global_crop),
                                        transforms.ToTensor()(w_local_crop1),
                                        transforms.ToTensor()(w_local_crop2)], dim=0)

                        img_s = torch.cat([transforms.ToTensor()(s_global_crop),
                                        transforms.ToTensor()(s_local_crop1),
                                        transforms.ToTensor()(s_local_crop2)], dim=0)

                        return idx, img_w, img_s                    
                    else:
                        return idx, img_w, self.strong_transform(img)
                elif self.alg in ['softmatch', 'freematch', 'freematch_entropy']:
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'pimodel':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'pseudolabel':
                    return idx, img_w
                elif self.alg == 'vat':
                    return idx, img_w
                elif self.alg == 'meanteacher':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'uda':
                    return idx, img_w, self.strong_transform(img)
                elif self.alg == 'mixmatch':
                    return idx, img_w, self.transform(img)
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return idx, img_w, img_s1, img_s2, img_s1_rot, rotate_v_list.index(rotate_v1)
                elif self.alg == 'fullysupervised':
                    return idx

    def __len__(self):
        return len(self.data)
