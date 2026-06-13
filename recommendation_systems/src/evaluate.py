import pandas as pd
import numpy as np
from tqdm import tqdm


def evaluate_recommender(full_anime_df, test_data, train_data, model, model_name, results_df, k=5, **kwargs):
  """Функция для подсчета метрик Precision@K и Recall@K, MAP@K и NDCG@K."""

  # Словари для сохранения метрик по пользователю
  hit_rate_for_user = {}
  precision_for_user = {}
  recall_for_user = {}
  ap_for_user = {}
  ndcg_for_user = {}

  # Формирование словаря из ID пользователя и, просмотренных им тайтлов из test
  user_ground_truth_dict = test_data.groupby('user_id')['anime_id'].apply(set).to_dict()
  users_to_evaluate = list(user_ground_truth_dict.keys())

  # Инициализация progress bar
  progress_bar = tqdm(users_to_evaluate, desc=f"Валидация {model_name}")

  for user_id in progress_bar:

    # Получение просмотренных пользователем тайтлов
    user_ground_truth = user_ground_truth_dict[user_id]

    # Удаление пользователя из словаря при отсутствии его релевантных тайтлов в test
    if not user_ground_truth:
      del user_ground_truth_dict[user_id]
      continue

    # Получение рекомендаций для пользователя
    try:
      recommended_ids = model.recommend(
        train_data=train_data,
        user_id=user_id,
        n=k,
        only_id=True,
        **kwargs
      )
    # Вывод в случае ошибки
    except Exception as e:
      print(f"Ошибка при вызове модели {model_name} для пользователя {user_id}: {e}")
      continue

    # Сохранение нулевых значений при отсутствии предсказания модели
    if not recommended_ids:
      hit_rate_for_user[user_id] = 0.0
      precision_for_user[user_id] = 0.0
      recall_for_user[user_id] = 0.0
      ap_for_user[user_id] = 0.0
      ndcg_for_user[user_id] = 0.0
      continue

    # Переменные для счетчика
    hits = 0
    sum_precs = 0.0
    dcg = 0.0

    # Ранжированный перебор рекомендаций модели
    for rank, anime_id in enumerate(recommended_ids, start=1):
      # Условие совпадения рекомендации со списком просмотренных аниме из test
      if anime_id in user_ground_truth:
        # Счетчик совпадений
        hits += 1
        # Расчет совпадения по рангу для AP
        sum_precs += hits / rank
        # Расчет совпадения по рангу для DCG
        dcg += 1.0 / np.log2(rank + 1)

    # Расчет hit_rate конкретного пользователя
    hit_rate_for_user[user_id] = int(hits > 0)

    # Расчет precision конкретного пользователя
    precision_for_user[user_id] = hits / len(recommended_ids)

    # Расчет recall конкретного пользователя
    recall_for_user[user_id] = hits / len(user_ground_truth)

    # Расчет ap конкретного пользователя
    ap_for_user[user_id] = sum_precs / min(len(user_ground_truth), k)

    # Расчет ndcg конкретного пользователя
    ideal_hits = min(len(user_ground_truth), k)
    idcg = sum([1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1)])
    ndcg_for_user[user_id] = dcg / idcg if idcg > 0 else 0.0

  # Подсчет количества пользователей
  evaluated_count = len(precision_for_user)
  if evaluated_count == 0:
    print(f"Ни один пользователь не был успешно оценен для {model_name}")
    return results_df

  # Расчет средних метрик
  hit_rate = sum(hit_rate_for_user.values()) / evaluated_count
  mean_precision = sum(precision_for_user.values()) / evaluated_count
  mean_recall = sum(recall_for_user.values()) / evaluated_count
  mean_ap = sum(ap_for_user.values()) / evaluated_count
  mean_ndcg = sum(ndcg_for_user.values()) / evaluated_count

  # Запись результатов в датафрейм
  new_row = pd.DataFrame([{
    'Модель': model_name,
    f'Hit Rate@{k}': hit_rate,
    f'Precision@{k}': mean_precision,
    f'Recall@{k}': mean_recall,
    f'MAP@{k}': mean_ap,
    f'NDCG@{k}': mean_ndcg
  }])
  results_df = pd.concat([results_df, new_row], ignore_index=True)
  return results_df

def get_best_alpha(full_anime_df, test_data, train_data, model, k=5):
  """Функция подсчета метрик гибридной модели при разных alpha"""

  # Инициализация датафрейма для сохранения метрик
  df = pd.DataFrame()

  # Формировка подвыборок
  sample_users = np.random.choice(test_data.user_id.unique(), size=500, replace=False)
  test_data = test_data[test_data.user_id.isin(sample_users)]

  # Подсчет метрик гибридной модели при разных alpha
  for alpha in np.arange(0.0, 1.1, 0.1):
    df = evaluate_recommender(  full_anime_df, test_data, train_data, model,
                    f'Hybrid_system_{alpha:.1f}', df, k=k, alpha=round(alpha, 1)
                              )
  return df