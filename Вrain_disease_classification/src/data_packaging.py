from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Протокол изменения и аугментации изображений из train
train_transforms = A.Compose([
    # Изменение размерности
    A.Resize(256, 256),

    # Аугментации
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=(15, -15), p=0.5),
    A.RandomBrightnessContrast(p=0.2),

    # Нормализация
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    # Преобразование в тензор
    A.pytorch.ToTensorV2()
    ])


# Протокол изменения изображений из val и test
test_transforms = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    A.pytorch.ToTensorV2()
    ])



def split_data(paths_df):
    """Функция, разбивающая данные на train, val и test, используя StratifiedGroupKFold"""

    # Разбиваем данные на train и val_test, не допуская распространения изображения с одним id на обе подвыборки
    splitter_1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_ids, val_test_ids = next(splitter_1.split(X=paths_df, y=paths_df['class'], groups=paths_df['image_id']))

    val_test_paths = paths_df.iloc[val_test_ids].reset_index(drop=True)

    # Разбиваем данные на val и test
    splitter_2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_ids, test_ids = next(splitter_2.split(X=val_test_paths, y=val_test_paths['class'], groups=val_test_paths['image_id']))

    train_paths = paths_df.iloc[train_ids].reset_index(drop=True)
    val_paths = val_test_paths.iloc[val_ids].reset_index(drop=True)
    test_paths = val_test_paths.iloc[test_ids].reset_index(drop=True)

    return train_paths, val_paths, test_paths



class LoadMRImage(Dataset):
    """Класс, автоматически загружающий изображения и добавляющий к ним метку класса"""

    def __init__(self, images_paths, transforms_type):
        self.labels = {'aneurysm': 0, 'cancer': 1, 'tumor': 2}
        self.images_paths = images_paths
        self.transforms_type = transforms_type

    def __getitem__(self, idx):
        image_row = self.images_paths.iloc[idx]

        img = cv2.imdecode(np.fromfile(image_row['full_path'], dtype=np.uint8), cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                      # Меняем порядок цветовых каналов

        transformed = self.transforms_type(image=img)                   # Производим трансформацию изображений
        img_tensor = transformed['image']

        label = self.labels[image_row['class']]                         # Получаем метку класса

        return img_tensor, label

    def __len__(self):
        return len(self.images_paths)



def get_datasets(train_paths, val_paths, test_paths):
    """Функция создания объектов DataLoader для всех подвыборок"""

    train_dataset = LoadMRImage(train_paths, train_transforms)
    val_dataset = LoadMRImage(val_paths, test_transforms)
    test_dataset = LoadMRImage(test_paths, test_transforms)

    return train_dataset, val_dataset, test_dataset



def denormalize_image(img_tensor):
    """Восстанавливает изображение после нормализации для визуализации"""

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)

    return img



def show_augmented_images(df, dataset):
    """Функция, визуализирующая исходное изображение и несколько его вариантов со случайными аугментациями"""

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    image_row = df.iloc[0]
    img = cv2.imdecode(np.fromfile(image_row['full_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Исходное изображение')
    ax[0, 0].axis('off')

    for axes in ax.flat[1:]:
        denorm_img = denormalize_image(dataset[0][0])
        axes.imshow(denorm_img.permute(1, 2, 0).numpy())
        axes.set_title('Пример аугментированного изображения')
        axes.axis('off')



def get_loaders(train_dataset, val_dataset, test_dataset):
    """Функция упаковки объектов DataLoader в батчи"""

    train = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val = DataLoader(val_dataset, batch_size=8)
    test = DataLoader(test_dataset, batch_size=8)

    return train, val, test