import torch
import os
from src.models import get_ssd, get_faster_rcnn, get_retinanet, get_fcos
from src.engine import train_model
from src.dataset import create_dataloader
from data_preparation import train_dataset, val_dataset

# Автоматический выбор устройства для вычислений
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.makedirs("models", exist_ok=True)

# Карта с функциями инициализаций моделей и, подходящими для моделей, гиперпараметрами
experiments = {
    "SSD300": {
        "model_fn": lambda: get_ssd(num_classes=4),
        "lr": 0.001,
        "batch_size": 8
    },
    "FasterRCNN_v2": {
        "model_fn": lambda: get_faster_rcnn(num_classes=4),
        "lr": 0.001,
        "batch_size": 4
    },
    "RetinaNet": {
        "model_fn": lambda: get_retinanet(num_classes=4),
        "lr": 0.001,
        "batch_size": 4
    },
    "FCOS": {
        "model_fn": lambda: get_fcos(num_classes=4),
        "lr": 0.0005,
        "batch_size": 8
    }
}

# Цикл автоматического обучения всех моделей
for name, config in experiments.items():
    print(f"\n" + "="*50)
    print(f" Запуск обучения архитектуры: {name}")
    print("="*50 + "\n")

    # Создание объектов DataLoader с подходящим для модели batch_size
    train_loader = create_dataloader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Инициализация модели
    model = config["model_fn"]().to(device)

    # Инициализация планировщика и оптимизатора с подходящим для модели learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5
    )

    # Цикл обучения модели
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=name,
        n_epochs=50
    )
