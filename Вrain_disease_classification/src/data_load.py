import kagglehub
import os
import numpy as np
import pandas as pd
import cv2


def get_data():
    """Функция, загружающая датасет"""

    # Загружаем данные
    path = kagglehub.dataset_download("shuvokumarbasakbd/brain-ct-medical-imaging-colorized-dataset")

    # Добавляем вложенные папки в общий путь
    path = os.path.join(path, 'Computed Tomography (CT) of the Brain/dataset')

    return path


def image_comparison(train_path, test_path):
    """Функция попиксельного сравнения изображений из train и test c одинаковыми названиями"""

    equal = 0
    not_equal = 0

    for root, dirs, files in os.walk(test_path):
        for filename in files:

            img1 = np.array(cv2.imread(os.path.join(root, filename)))
            img2 = np.array(cv2.imread(os.path.join(train_path, os.path.basename(root), filename)))

            if np.array_equal(img1, img2):
                equal += 1
            else:
                not_equal += 1

    print(f' Количество одинаковых изображений: {equal} \n Количество неодинаковых изображений: {not_equal}')


def get_paths_df(train_path):
    paths_list = []

    for root, dirs, files in os.walk(train_path):
        for filename in files:
            paths_list.append({
                'full_path': os.path.join(root, filename),
                'class': os.path.basename(root),
                'image_id': filename.split('_')[0]
            })

    paths_df = pd.DataFrame(paths_list)
    return paths_df