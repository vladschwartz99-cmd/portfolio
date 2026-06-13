import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD

def get_user_interactions_split(full_anime_df, data, user_id, only_rated=True):
    """Функция, формирования списков из ID аниме, которые пользователь посмотрел и не смотрел"""

    # Получение всех anime_id
    all_anime_ids = set(full_anime_df['anime_id'])

    # Получение ID тайтлов, которые пользователь посмотрел
    user_rated_anime_id = set(data[data['user_id'] == user_id]['anime_id'])

    if only_rated:
        return user_rated_anime_id
    else:
        # Получение ID тайтлов, которые пользователь не смотрел
        user_not_rated_anime_id = all_anime_ids - user_rated_anime_id
        return user_rated_anime_id, user_not_rated_anime_id


def cosine_similarity_matrix(full_anime_df):
    """Функция, формирующая матрицу косинусного сходства на основании TF-IDF"""

    # Построение матрицы TF-IDF по всему каталогу тайтлов
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(full_anime_df['for_tfidf'])

    # Подсчет матрицы косинусного сходства тайтлов между собой
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Преобразование матрицы косинусного сходства в датафрейм с индексами тайтлов по осям
    anime_ids = full_anime_df['anime_id'].values
    cosine_sim_df = pd.DataFrame(similarity_matrix, index=anime_ids, columns=anime_ids)

    return cosine_sim_df


def prepare_surprise_dataset(train_df):
    """Конвертация train в формат, подходящий для библиотеки surprise"""

    train_df_rated = train_df[train_df['rating'] != -1][['user_id', 'anime_id', 'rating']]

    # Упаковка данных в surprise Dataset
    reader = Reader(rating_scale=(1.0, 10.0))
    surprise_train_data = Dataset.load_from_df(train_df_rated, reader)
    surprise_train = surprise_train_data.build_full_trainset()

    return surprise_train


def train_svd(train_df, n_factors=20, random_state=42):
    """Функция обучения модели SVD из библиотеки surprise"""

    # Подготовка данных
    surprise_train = prepare_surprise_dataset(train_df)

    # Инициализация модели
    model = SVD(n_factors=n_factors, random_state=random_state)

    # Обучение модели
    model.fit(surprise_train)

    return model