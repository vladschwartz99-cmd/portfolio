import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

BOXES_LABEL_MAP = {1: 'WBC', 2: 'RBC', 3: 'Platelets'}
COLORS = {1: 'red', 2: 'green', 3: 'blue'}

def plot_image_with_boxes(image_with_boxes, predicted_boxes=None, ax=None, title=None):
    """Отрисовка одного изображения с истинными или предсказанными рамками"""
    image, target = image_with_boxes
    
    # Конвертация изображение из тензора PyTorch в формат NumPy
    img_to_show = image.permute(1, 2, 0).detach().cpu().numpy()
    boxes_dict = predicted_boxes if predicted_boxes is not None else target

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.imshow(img_to_show)
    if title:
        ax.set_title(title, fontsize=14)

    # Конвертация рамки из тензоров PyTorch в формат NumPy
    boxes = boxes_dict['boxes'].detach().cpu().numpy()
    labels = boxes_dict['labels'].detach().cpu().numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        
        label_name = BOXES_LABEL_MAP.get(label, 'Unknown')
        color = COLORS.get(label, 'yellow')

        # Отрисовка ограничивающей рамки
        rect = patches.Rectangle(
            (xmin, ymin), width, height, 
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Текст с названием класса
        ax.text(
            xmin, ymin, label_name, color='white',
            verticalalignment='top', 
            bbox={'color': color, 'pad': 2, 'alpha': 0.7}
        )
        
    ax.axis('off')

def save_dataset_collage(dataset, output_path="reports/figures/dataset_collage.png"):
    """Создание и сохранение коллажа из 16 случайных изображений датасета"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    samples = random.sample(range(len(dataset)), 16)

    for ax, sample in zip(axes.flatten(), samples):
        plot_image_with_boxes(dataset[sample], ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def predict_single_image(model, image_path, threshold, device):
    """Загрузка одного изображения, инференс и фильтрация по порогу"""
    # Пайплайн трансформации для предсказания
    transforms = A.Compose([
        A.Resize(300, 300),
        ToTensorV2()
    ])
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transforms(image=img)
    img_tensor = (transformed['image'] / 255.0).to(device)

    model.eval()
    with torch.no_grad():
        raw_prediction = model([img_tensor])[0]

    # Фильтрация по порогу
    mask = raw_prediction['scores'] > threshold
    prediction = {
        'boxes': raw_prediction['boxes'][mask],
        'labels': raw_prediction['labels'][mask],
        'scores': raw_prediction['scores'][mask]
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return prediction


def save_threshold_grid(model, image_path, dataset_sample, device, output_path):
    """Генерация и сохранение сетки предсказаний с порогами от 0.0 до 0.95"""
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    thresholds = np.arange(0, 1, 0.05)
    model_name = model.__class__.__name__

    for ax, threshold in zip(axes.flatten(), thresholds):
        pred = predict_single_image(model, image_path, threshold, device)
        
        # Визуализация предсказания поверх исходного изображения из датасета
        plot_image_with_boxes(dataset_sample, predicted_boxes=pred, ax=ax)
        ax.set_title(f'{model_name} | Conf: {threshold:.2f}', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def visualize_final_comparison(models_dict, image_path, dataset_sample, device, output_path=None):
    """Создание коллажа 2х2 для сравнения работы четырех моделей на одном снимке"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for ax, (model_name, (model, threshold)) in zip(axes.flatten(), models_dict.items()):
        # Генерация предсказания модели
        pred = predict_single_image(model, image_path, threshold, device)
        
        # Отрисовка предсказание поверх исходного изображения из датасета
        plot_image_with_boxes(dataset_sample, predicted_boxes=pred, ax=ax)
        ax.set_title(f"{model_name} (Threshold: {threshold})", fontsize=14)
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close()
