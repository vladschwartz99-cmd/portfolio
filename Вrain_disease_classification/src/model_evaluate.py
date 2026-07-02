import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.data_packaging import denormalize_image



# Устройство для вычислений
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def get_predictions(model, dataloader):
  """Функция получения списков из предсказаний моделей и истинных меток"""

  model.eval()

  y_true = []
  y_pred = []

  with torch.no_grad():
      for X_batch, y_batch in dataloader:

          # Перевод изображений на устройство
          X_batch = X_batch.to(device)

          # Получение предсказания модели
          logits = model(X_batch)
          preds = logits.argmax(dim=1).cpu().numpy()

          # Добавление элементов в списки
          y_true.extend(y_batch.numpy())
          y_pred.extend(preds)

  return y_true, y_pred



def get_reports(model, dataloader):
    """Функция, выводящая classification_report и confusion_matrix для модели"""

    # Получение предсказаний для test и истинных меток
    y_true, y_pred = get_predictions(model, dataloader)

    # Вывод отчетов
    print('='*35)
    print('Метрики по классам')
    print('='*35)

    class_names = ['aneurysm', 'cancer', 'tumor']
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print('='*35)
    print('Матрица ошибок')
    print('='*35)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap='Blues')



def visualize_gradcam(model, layers, dataset, ids):
    """Функция, выводящая исходные изображения и тепловые карты значимости областей для классификации моделью"""

    # Разморозка градиентов последного сверточного блока
    for param in layers.parameters():
        param.requires_grad = True

    labels = {0: 'aneurysm', 1: 'cancer', 2: 'tumor'}

    model.eval()

    # Инициализация GradCAM
    cam = GradCAM(model=model, target_layers=[layers])

    # Проход по всем индексам изображений
    for idx in ids:
        # Получение изображения и его метки класса
        img_tensor, label = dataset[idx]

        # Перевод изображения на устройство и изменение размерности
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Получение маски для heatmap
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        # Перевод изображения в формат для визуализации
        rgb_img = denormalize_image(img_tensor)
        rgb_img = rgb_img.permute(1, 2, 0).numpy()

        # Наложение маски на изображение
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Получение предсказания модели
        pred = model(input_tensor).argmax(1).item()

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

        # Вывод исходного изображения
        ax[0].imshow(rgb_img)
        ax[0].set_title(f'True: {labels[label]}')
        ax[0].axis('off')

        # Вывод heatmap
        ax[1].imshow(visualization)
        ax[1].set_title(f'Pred: {labels[pred]}')
        ax[1].axis('off')



def get_model_errors(model, dataset, n_errors=5):
    """Функция, выводящая индексы 5 изображений из test в классификации которых модель ошибается"""

    y_true, y_pred = get_predictions(model, dataset)
    diff_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if pred != true]

    return diff_indices[:n_errors]