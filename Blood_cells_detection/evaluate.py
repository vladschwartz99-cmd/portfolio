import os
import torch
import pandas as pd
from data_preparation import test_paths, test_dataset
from src.dataset import create_dataloader
from src.models import get_ssd, get_faster_rcnn, get_retinanet, get_fcos
from src.engine import calculate_map
from src.visualization import save_threshold_grid, visualize_final_comparison

# Автоматический выбор устройства для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("reports", exist_ok=True)

# Загрузка тестовых данных
test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)

# Карта моделей и путей к их весам, сохраненным при обучении
models_config = {
    'SSD': (get_ssd, 'models/best_SSD300.pth'),
    'FRCNN': (get_faster_rcnn, 'models/best_FasterRCNN_v2.pth'),
    'RetinaNet': (get_retinanet, 'models/best_RetinaNet.pth'),
    'FCOS': (get_fcos, 'models/best_FCOS.pth')
}

# Словарь для сохранения моделей с кастомными весами
loaded_models = {}

# Цикл инициализации моделей с кастомными весами
for name, (model_fn, weight_path) in models_config.items():
    model = model_fn(num_classes=4, pretrained=False)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        loaded_models[f'{name}_model'] = model
    else:
        print(f" Предупреждение: Файл весов {weight_path} не найден! Пропуск.")
        continue


# Словарь для сохранения метрик
results = {}

# Цикл подсчета mAP для моделей
for name, model in loaded_models.items():
    res_metrics = calculate_map(model, test_loader, device, class_metrics=True)
    results[name] = res_metrics

# Сборка финального датафрейма метрик
data_matrix = {}
for model_name, res in results.items():
    data_matrix[model_name] = [
        res['map'].item(),
        res['map_50'].item(),
        res['map_75'].item(),
        *res['map_per_class'].tolist()
    ]

index_labels = ['mAP', 'mAP@50', 'mAP@75', 'mAP WBC (Лейкоциты)', 'mAP RBC (Эритроциты)', 'mAP Platelets (Тромбоциты)']
df_results = pd.DataFrame(data_matrix, index=index_labels)
df_results = df_results.round(4)

# Сохранение отчета
df_results.to_csv("reports/test_metrics.csv")
df_results.to_markdown("reports/test_metrics.md")

# Создание директории для отчетных изображений
os.makedirs("reports/figures/models_thresholds", exist_ok=True)

# Цикл по созданию и сохранению сетки предсказаний с разными порогами для всех моделей
for name, model in loaded_models.items():
    save_threshold_grid(model, test_paths[0], test_dataset[0], device, f"reports/figures/models_thresholds/{name}_thresholds.png")

# Карта моделей и лучших порогов, определенных в ходе тестов
best_configs = {
    'SSD300': (loaded_models['SSD_model'], 0.1),
    'Faster R-CNN': (loaded_models['FRCNN_model'], 0.05),
    'RetinaNet': (loaded_models['RetinaNet_model'], 0.3),
    'FCOS': (loaded_models['FCOS_model'], 0.4)
}

# Создание директории для отчетных изображений
os.makedirs("reports/figures/comparison_of_models", exist_ok=True)

# Генерация финальных отчетных изображений
for idx in [0, 25, 50, 70]:
    visualize_final_comparison(
        best_configs, 
        test_paths[idx], 
        test_dataset[idx], 
        device, 
        output_path=f"reports/figures/comparison_of_models/comparison_sample_{idx}.png"
    )
