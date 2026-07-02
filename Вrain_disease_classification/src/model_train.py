import torch
import torchvision as tv
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import clear_output



# Устройство для вычислений
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def plot_learning_curves(history):
    """Функция для вывода графиков лосса и accuracy во время обучения"""

    fig = plt.figure(figsize=(20, 7))

    plt.subplot(1,2,1)
    plt.title('Loss', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epoch_number', fontsize=15)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Accuracy', fontsize=15)
    plt.plot(history['acc']['train'], label='train')
    plt.plot(history['acc']['val'], label='val')
    plt.ylabel('accuracy', fontsize=15)
    plt.xlabel('epoch_number', fontsize=15)
    plt.legend()
    plt.show()



def train_model(model, criterion,
                optimizer, train_batch_gen,
                val_batch_gen, num_epochs=50, patience=5):
    """Функция для обучения модели и вывода метрик во время обучения"""

    # Формирование словаря, для сохранения метрик между эпохами
    history = defaultdict(lambda: defaultdict(list))

    # Переменные для преждевременной остановки
    best_val_loss = float('inf')
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        # Переменные для метрик
        train_correct = 0
        train_total = 0
        train_loss = 0
        val_correct = 0
        val_total = 0
        val_loss = 0

        start_time = time.time()

        model.train()

        # Проход по train
        for X_batch, y_batch in tqdm.tqdm_notebook(train_batch_gen):

            # Перевод данных на устройство
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Обнуление градиента
            optimizer.zero_grad()

            # Получение ответа модели
            logits = model(X_batch)

            # Подсчет лосса
            loss = criterion(logits, y_batch.long().to(device))

            # Расчет градиента и шаг оптимизатора
            loss.backward()
            optimizer.step()

            # Расчет метрик по батчам
            train_loss += loss.item()
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_correct += (y_pred == y_batch.cpu()).sum().item()
            train_total += len(y_batch)

        # Расчет метрик для всего train + сохранение в "историю"
        train_loss /= len(train_batch_gen)
        train_acc = train_correct / train_total
        history['loss']['train'].append(train_loss)
        history['acc']['train'].append(train_acc)


        model.eval()

        # Проход по val
        with torch.no_grad():

            for X_batch, y_batch in tqdm.tqdm_notebook(val_batch_gen):

                # Перевод данных на устройство
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Получение ответа модели
                logits = model(X_batch)

                # Подсчет лосса
                loss = criterion(logits, y_batch.long().to(device))

                # Расчет метрик по батсчам
                val_loss += loss.item()
                y_pred = logits.max(1)[1].detach().cpu().numpy()
                val_correct += (y_pred == y_batch.cpu()).sum().item()
                val_total += len(y_batch)

        # Расчет метрик для всего val + сохранение в "историю"
        val_loss /= len(val_batch_gen)
        val_acc = val_correct / val_total
        history['loss']['val'].append(val_loss)
        history['acc']['val'].append(val_acc)

        # Проверка модели на улучшение val_loss
        if val_loss < best_val_loss:

            # Пересохранение лучшей метрики
            best_val_loss = val_loss

            # Сохранение весов модели
            best_weights = model.state_dict()

            # Обнуление счетчика эпох без улучшения
            epochs_without_improvement = 0

        # Шаг счетчика при отсутствии улучшения
        else:
            epochs_without_improvement += 1

        # Ранняя остановка
        if epochs_without_improvement >= patience:
            break

        # Избегание захломления вывода функции обучения
        clear_output()

        # Вывод метрик эпохи обучения
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))

        # Вывод кривых обучения по сохраненным значениям метрик
        plot_learning_curves(history)

    # Загрузка лучших весов в модель
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history



def get_resnet():
    """Функция, инициализации предобученной модели ResNet18 с оптимизатором и функцией потерь"""

    # Загрузка модели с исходными весами
    weights = tv.models.ResNet18_Weights.DEFAULT
    resnet = tv.models.resnet18(weights=weights)

    # Заморозка слоев
    for param in resnet.parameters():
        param.requires_grad = False

    # Добавление dropout и классификатора
    resnet.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(512, 3)
    )

    # Перенос модели на устройство
    resnet = resnet.to(device)

    # Инициализация функции потерь и оптимизатора
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

    return resnet, criterion, optimizer



def get_vgg():
    """Функция, инициализации предобученной модели VGG16 с оптимизатором и функцией потерь"""

    # Загрузка модели с исходными весами
    weights = tv.models.VGG16_Weights.DEFAULT
    vgg = tv.models.vgg16(weights=weights)

    # Заморозка слоев
    for param in vgg.parameters():
        param.requires_grad = False

    # Добавление классификатора (dropout в модели уже есть)
    vgg.classifier[6] = torch.nn.Linear(4096, 3)

    # Перенос модели на устройство
    vgg = vgg.to(device)

    # Инициализация функции потерь и оптимизатора
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg.parameters(), lr=0.001)

    return vgg, criterion, optimizer



def get_efficientnet():
    """Функция, инициализации предобученной модели EfficientNet_B0 с оптимизатором и функцией потерь"""

    # Загрузка модели с исходными весами
    weights = tv.models.EfficientNet_B0_Weights.DEFAULT
    efficientnet = tv.models.efficientnet_b0(weights=weights)

    # Заморозка слоев
    for param in efficientnet.parameters():
        param.requires_grad = False

    # Добавление классификатора (dropout в этой модели тоже уже есть)
    efficientnet.classifier[1] = torch.nn.Linear(1280, 3)

    # Перенос модели на устройство
    efficientnet = efficientnet.to(device)

    # Инициализация функции потерь и оптимизатора
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(efficientnet.parameters(), lr=0.001)

    return efficientnet, criterion, optimizer