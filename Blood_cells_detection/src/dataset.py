import os
import xml.etree.ElementTree as ET
import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

CLASS_LABELS = {'background': 0, 'WBC': 1, 'RBC': 2, 'Platelets': 3}

def get_transforms(train=True, img_size=300):
    """Функция, возвращающая пайплайн трансформаций изображений"""
    if train:
        return A.Compose([
              A.Resize(img_size, img_size),
              A.HorizontalFlip(p=0.5),
              A.RandomBrightnessContrast(p=0.2),
              ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))
    else:
        return A.Compose([
              A.Resize(img_size, img_size),
              ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))

class BCCDDataset(Dataset):
    """Класс для загрузки и препроцессинга изображений"""
    def __init__(self, images_paths, annotations_dir, transforms):
        self.images_paths = images_paths
        self.annotations_dir = annotations_dir
        self.transforms = transforms

    def _parse_boxes(self, annotation_path):
        """Парсинг XML-аннотаций"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASS_LABELS:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Валидация площади рамки
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASS_LABELS[name])

        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        annotation_path = os.path.join(self.annotations_dir, f'{basename}.xml')
        
        boxes, labels = self._parse_boxes(annotation_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
        img_tensor = transformed['image'] / 255.0  # Нормализация [0, 1]

        target = {
            'boxes': torch.as_tensor(transformed['bboxes'], dtype=torch.float32),
            'labels': torch.as_tensor(transformed['class_labels'], dtype=torch.int64)
        }

        return img_tensor, target

    def __len__(self):
        return len(self.images_paths)

def collate_fn(batch):
    """Группировка батча для детекции"""
    return tuple(zip(*batch))

def create_dataloader(dataset, batch_size=8, shuffle=False):
    """Сборка DataLoader"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
