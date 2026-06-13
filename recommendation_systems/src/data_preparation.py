import pandas as pd

def preprocessing_for_eda(df_anime, df_rating):
    """Функция предобработки данных для EDA"""

    # Копирование датафрейма для защиты исходных данных
    df_anime = df_anime.copy()

    # Переименование столбца с рейтингом для последующего объединения датафреймов
    df_anime = df_anime.rename(columns={'rating': 'mean_rating_anime'})

    # Объединение датафреймов и удаление возможных повторений
    df = df_rating.merge(df_anime, on='anime_id')
    df = df.drop_duplicates()

    return df

def preprocessing_anime(df):
    """Функция предобработки данных об аниме"""

    # Копирование датафрейма для защиты исходных данных
    df = df.copy()

    # Замена некорректных значений
    df['episodes'] = df['episodes'].replace('Unknown', None)

    # Заполнение пропусков и изменение типа данных
    df['episodes'] = df['episodes'].fillna(df['episodes'].mode()[0])
    df['episodes'] = pd.to_numeric(df['episodes'], downcast='integer', errors='coerce')
    df = df.dropna()

    # Разделение единого текстового поля с жанрами
    df['genre'] = df['genre'].str.replace(', ', ' ')

    # Добавление признака для последующего TF-IDF
    df['for_tfidf'] = df['genre'] + ' ' + df['type']

    # Переименование столбца с рейтингом для последующего объединения датафреймов
    df.rename(columns={'rating': 'mean_rating_anime'}, inplace=True)

    return df

def data_fusion(df_anime, df_rating):
    """Функция объединения датафреймов и последующего удаления пропусков и повторений"""

    df = df_rating.merge(df_anime, on='anime_id')
    df = df.dropna()
    df = df.drop_duplicates()

    return df

def add_attributes(df, attributes, global_mean, global_median):
    """Функция, присоединяющая к датафрейму новые признаки, заполняя возможные пропуски глобальным средним"""

    df = df.merge(attributes, on='user_id', how='left')
    df['mean_rating_user'] = df['mean_rating_user'].fillna(global_mean)
    df['median_rating_user'] = df['median_rating_user'].fillna(global_median)

    return df

def train_test_processing(df_anime, df_rating, rating_threshold=7, random_state=42):
    """Функция по предобработке данных и их разбиению на train и test"""

    #Предобработка данных об аниме
    df_anime = preprocessing_anime(df_anime)

    # Объединение данных
    df = data_fusion(df_anime, df_rating)

    # Leave-10-Out разделение данных на train и test
    train_parts = []
    test_parts = []

    # Цикл, перебирающий датафрейм по пользователю
    for user_id, user_df in df.groupby("user_id"):

        # Формирование датафрейма только с релевантными для пользователя тайтлами
        relevant = user_df[
            user_df["rating"] >= rating_threshold
            ]

        # Отсеивание пользователей с низким количеством оценок
        if len(relevant) < 15:
            continue

        # Формирование train и test подвыборок для пользователя
        test_df = relevant.sample(
            n=10,
            random_state=random_state
        )

        train_df = user_df.drop(test_df.index)

        train_parts.append(train_df)
        test_parts.append(test_df)

    # Формирование итоговых train и test подвыборок
    train, test = pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)

    # Формирование новых признаков пользователей на основе train
    train_df_rated = train[train.rating != -1]
    user_stats = train_df_rated.groupby('user_id')['rating'].agg(['mean', 'median']).reset_index()
    user_stats.columns = ['user_id', 'mean_rating_user', 'median_rating_user']

    # Подсчет глобальных средних для заполнения возможных пропусков
    global_mean = train_df_rated['rating'].mean()
    global_median = train_df_rated['rating'].median()

    # Добавление новых признаков (подсчитанных на train) к train и test
    train = add_attributes(train, user_stats, global_mean, global_median)
    test = add_attributes(test, user_stats, global_mean, global_median)

    return train, test