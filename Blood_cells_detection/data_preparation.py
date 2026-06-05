import os
from src.dataset import get_transforms, BCCDDataset
from src.visualization import save_dataset_collage

BASE_DIR = './BCCD_dataset/BCCD/ImageSets/Main/'
DATA_DIR = './BCCD_dataset/BCCD/JPEGImages'
ANNOT_DIR = './BCCD_dataset/BCCD/Annotations'

def get_image_paths(split_file):
    with open(os.path.join(BASE_DIR, split_file), 'r') as f:
        names = f.read().splitlines()
    return [os.path.join(DATA_DIR, f'{name}.jpg') for name in names]

# Сбор путей
train_paths = get_image_paths('train.txt')
val_paths = get_image_paths('val.txt')
test_paths = get_image_paths('test.txt')

# Создание объектов BCCDDataset
train_dataset = BCCDDataset(train_paths, ANNOT_DIR, get_transforms(train=True))
val_dataset = BCCDDataset(val_paths, ANNOT_DIR, get_transforms(train=False))
test_dataset = BCCDDataset(test_paths, ANNOT_DIR, get_transforms(train=False))

# Создание директории для отчетных изображений
os.makedirs("reports/figures/initial_segmentation", exist_ok=True)

# Генерация и сохранение коллажей изображений с исходной разметкой
save_dataset_collage(train_dataset, "reports/figures/initial_segmentation/train_collage.png")
save_dataset_collage(val_dataset, "reports/figures/initial_segmentation/val_collage.png")
save_dataset_collage(test_dataset, "reports/figures/initial_segmentation/test_collage.png")