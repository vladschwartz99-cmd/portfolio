import pandas as pd
import numpy as np
from src.model_utils import get_user_interactions_split, cosine_similarity_matrix, train_svd


class Baseline:
    """Рекомендательная система, рекомендующая n самых популярных тайтлов
        из обучающей выборки, которые пользователь еще не смотрел"""

    def __init__(self, n_top_cached=500):
        self.n_top_cached = n_top_cached
        self.full_anime_df = None
        self.top_anime_df = None

    def fit(self, full_anime_df):
        """Вычисление топа популярных тайтлов"""

        self.full_anime_df = full_anime_df.copy()

        self.top_anime_df = full_anime_df.nlargest(self.n_top_cached, 'members')[['anime_id', 'name']]
        return self

    def recommend(self, train_data, user_id, n=5, only_id=True):
        """Генерация рекомендаций для конкретного пользователя"""

        if self.top_anime_df is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод .fit()")

        # Получение ID тайтлов, которые пользователь посмотрел
        rated_anime_id = get_user_interactions_split(self.full_anime_df, train_data, user_id, only_rated=True)

        # Формирование списка из непросмотренных пользователем тайтлов, упорядоченный по популярности
        top_for_user = [row for row in self.top_anime_df.itertuples(index=False) if row.anime_id not in rated_anime_id][:n]
        top_for_user = pd.DataFrame(top_for_user)
        top_id_for_user = top_for_user['anime_id'].tolist()

        # Вывод пустого списка при отсутствии рекомендаций
        if not top_id_for_user:
            return []

        # Вывод датафрейма или списка из ID тайтлов
        if only_id:
            return top_id_for_user

        recommended_anime = self.full_anime_df.set_index('anime_id').loc[top_id_for_user].reset_index()
        return recommended_anime



class ContentBased:
    """Рекомендательная система, рекомендующая n тайтлов, основываясь
         на их близости к уже понравившимся пользователю аниме"""

    def __init__(self):
        self.full_anime_df = None
        self.cosine_sim_df = None
        self.baseline = Baseline()

    def fit(self, full_anime_df):
        """Обучение модели: расчет TF-IDF и матрицы сходства"""

        self.full_anime_df = full_anime_df.copy()

        # Формирование матрицы косинусного сходства на основании TF-IDF
        self.cosine_sim_df = cosine_similarity_matrix(self.full_anime_df)

        # Обучение базовой модели для вывода альтернативных рекомендаций
        self.baseline.fit(self.full_anime_df)
        return self

    def recommend(self, train_data,user_id, n=5, only_id=True):
        """Генерация рекомендаций для конкретного пользователя"""

        if self.cosine_sim_df is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод .fit()")

        # Получение ID тайтлов, которые пользователь посмотрел и не смотрел
        rated_anime_id, not_rated_anime_id = get_user_interactions_split(self.full_anime_df, train_data, user_id, only_rated=False)

        # Формирование списков ID релевантных тайтлов и их оценок
        user_logs = train_data[(train_data['user_id'] == user_id) & (train_data['rating'] >= 7)].sort_values('rating', ascending=False)
        relevant_anime = user_logs['anime_id'].tolist()
        weights = user_logs['rating'].tolist()

        # Вывод самых популярных тайтлов, в случае отсутствия релевантных аниме у пользователя в train
        if len(relevant_anime) == 0:
            return self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

        # Получение матрицы сходства для релевантных тайтлов пользователя
        sub_sim_matrix = self.cosine_sim_df.loc[:, relevant_anime].values

        # Вычисление взвешенного среднего сходства каждого тайтла с релевантными тайтлами пользователя
        all_content_scores = np.average(sub_sim_matrix, axis=1, weights=weights)

        # Сопоставление ID тайтлов с их средним сходством с релевантными тайтлами пользователя
        content_scores_series = pd.Series(all_content_scores, index=self.cosine_sim_df.index)

        # Фильтрация от просмотренных тайтлов
        valid_not_rated = content_scores_series.index.intersection(not_rated_anime_id)
        filtered_scores = content_scores_series[valid_not_rated]

        # Извлечение n самых похожих тайтлов
        final_recommendations = filtered_scores.nlargest(n).index.tolist()

        # Добавление к рекомендациям самых популярных тайтлов при их нехватке или отсутствии
        if len(final_recommendations) < n:

            # Получение списка ID самых популярных тайтлов
            baseline_recommendations = self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

            # Добавление не повторяющихся тайтлов
            for anime_id in baseline_recommendations:
                if anime_id not in final_recommendations:
                    final_recommendations.append(anime_id)

                if len(final_recommendations) >= n:
                    break

        # Вывод датафрейма или списка из ID тайтлов
        if only_id:
            return final_recommendations

        recommended_anime = self.full_anime_df.set_index('anime_id').loc[final_recommendations].reset_index()
        return recommended_anime



class Collaborative:
    """Рекомендательная система, рекомендующая n тайтлов для пользователя,
            основываясь на его похожести на других пользователей"""

    def __init__(self):
        self.full_anime_df = None
        self.model = None
        self.baseline = Baseline()

    def fit(self, full_anime_df, train_data):
        """Обучение модели: обучение модели SVD"""

        self.full_anime_df = full_anime_df.copy()

        # Инициализация и обучение модели SVD
        self.model = train_svd(train_data)

        # Обучение базовой модели для вывода альтернативных рекомендаций
        self.baseline.fit(full_anime_df)

        return self

    def recommend(self, train_data, user_id, n=5, only_id=True):
        """Генерация рекомендаций для конкретного пользователя"""

        if self.model is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод .fit()")

        # Получение списков ID тайтлов, просмотренных и непросмотренных пользователем
        rated_anime_id, not_rated_anime_id = get_user_interactions_split(self.full_anime_df, train_data, user_id, only_rated=False)

        # Вывод самых популярных тайтлов, в случае отсутствия непросмотренных аниме
        if len(not_rated_anime_id) == 0:
            return self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

        # Формировка датасета для загрузки в обученную модель
        unwatched = [(user_id, anime_id, 0) for anime_id in not_rated_anime_id]

        # Получение предсказаний модели для непросмотренных тайтлов
        predictions = self.model.test(unwatched)

        # Сортировка рекомендаций по предсказанным оценкам
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Взятие топ-N тайтлов
        final_recommendations = [pred.iid for pred in predictions[:n]]

        # Добавление к рекомендациям самых популярных тайтлов при их нехватке или отсутствии
        if len(final_recommendations) < n:

            # Получение списка ID самых популярных тайтлов
            baseline_recommendations = self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

            # Добавление не повторяющихся тайтлов
            for anime_id in baseline_recommendations:
                if anime_id not in final_recommendations:
                    final_recommendations.append(anime_id)

                if len(final_recommendations) >= n:
                    break

        # Вывод результата
        if only_id:
            return final_recommendations

        recommended_anime = self.full_anime_df.set_index('anime_id').loc[final_recommendations].reset_index()
        return recommended_anime



class Hybrid:
    def __init__(self):
        self.full_anime_df = None
        self.cosine_sim_df = None
        self.model = None
        self.baseline = Baseline()

    def fit(self, full_anime_df, train_data):
        """Обучение модели: расчет TF-IDF, матрицы сходства и обучение модели SVD"""

        self.full_anime_df = full_anime_df.copy()

        # Формирование матрицы косинусного сходства на основании TF-IDF
        self.cosine_sim_df = cosine_similarity_matrix(self.full_anime_df)

        # Инициализация и обучение модели SVD
        self.model = train_svd(train_data)

        # Обучение базовой модели для вывода альтернативных рекомендаций
        self.baseline.fit(self.full_anime_df)
        return self

    def recommend(self, train_data, user_id, n=5, alpha=0.3, only_id=True):
        """Генерация рекомендаций для конкретного пользователя"""

        if self.model is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод .fit()")

        # Получение списков ID тайтлов, просмотренных и не просмотренных пользователем
        rated_anime_id, not_rated_anime_id = get_user_interactions_split(self.full_anime_df, train_data, user_id, only_rated=False)

        # Вывод самых популярных тайтлов, в случае отсутствия не просмотренных
        if len(not_rated_anime_id) == 0:
            return self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

        # Формирование списков ID релевантных тайтлов и их оценок
        user_logs = train_data[(train_data['user_id'] == user_id) & (train_data['rating'] >= 7)].sort_values('rating', ascending=False)
        relevant_anime = user_logs['anime_id'].tolist()
        weights = user_logs['rating'].tolist()

        # Вывод самых популярных тайтлов, в случае отсутствия релевантных тайтлов у пользователя в train
        if len(relevant_anime) == 0:
            return self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

        # Получение матрицы сходства для релевантных тайтлов пользователя
        sub_sim_matrix = self.cosine_sim_df.loc[:, relevant_anime].values

        # Вычисление взвешенного среднего сходства каждого тайтла с релевантными тайтлами пользователя
        all_content_scores = np.average(sub_sim_matrix, axis=1, weights=weights)

        # Формирование словаря из ID тайтла и усредненного сходства
        content_scores_dict = dict(zip(self.cosine_sim_df.index, all_content_scores))

        predictions = []

        for anime_id in not_rated_anime_id:
            if anime_id not in content_scores_dict:
                continue

            # Получение предсказания модели коллаборативной фильтрации
            collaborative_score = self.model.predict(user_id, anime_id).est

            # Получение контентного счета для данного тайтла
            content_score = content_scores_dict[anime_id]

            # Вычисление итоговой оценки тайтла и добавление ее в список
            hybrid_score = (alpha * 10 * content_score) + ((1 - alpha) * collaborative_score)
            predictions.append((anime_id, hybrid_score))

        # Сортировка рекомендаций по предсказанным оценкам
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Взятие топ-N тайтлов
        final_recommendations = [anime_id for anime_id, _ in predictions[:n]]

        # Добавление к рекомендациям самых популярных тайтлов при их нехватке или отсутствии
        if len(final_recommendations) < n:

            # Получение списка ID самых популярных тайтлов
            baseline_recommendations = self.baseline.recommend(train_data, user_id, n=n, only_id=only_id)

            # Добавление не повторяющихся тайтлов
            for anime_id in baseline_recommendations:
                if anime_id not in final_recommendations:
                    final_recommendations.append(anime_id)

                if len(final_recommendations) >= n:
                    break

        # Вывод результата
        if only_id:
            return final_recommendations

        recommended_anime = self.full_anime_df.set_index('anime_id').loc[final_recommendations].reset_index()
        return recommended_anime