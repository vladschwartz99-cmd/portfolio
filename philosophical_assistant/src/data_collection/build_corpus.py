import re
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter



def read_text(path):
    """Функция чтения файла по его пути"""

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()



def clean_gutenberg(text):
    """Функция очистки тестов от служебной информации"""

    # Определение границ
    start_marker = '*** START OF'
    end_marker = '*** END OF'

    # Очистка начала текста
    start = text.find(start_marker)
    text = text[start:]

    # Очистка конца текста
    end = text.find(end_marker)
    text = text[:end]

    return text.strip()



def corpus_add_gutenberg(corpus, row):
    """Функция добавления в список текста с Gutenberg с его метаданными"""

    # Получение пути к тексту
    filepath = row['local_path']

    # Прочтение текста
    text = read_text(filepath)

    # Очистка текста
    text = clean_gutenberg(text)

    # Добавление данных в список
    corpus.append({ 'source': 'gutenberg', 'author': row['author'],
                    'title': row['title'], 'text': text
                    })



def corpus_add_sep(corpus):
    """Функция добавления в список текста SEP с его метаданными"""

    # Чтение файла с данными
    sep_df = pd.read_parquet('data/raw/sep/sep.parquet')

    # Добавление данных в список
    for _, row in sep_df.iterrows():
        corpus.append({ 'source': 'sep', 'author': None,
                        'title': row['category'], 'text': row['text']
                        })



def normalize_text(text):
    """Функция нормализации текста"""

    # Удаление переноса на новую строку и лишних пробелов
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    return text.strip()



def prepare_corpus(corpus):
    """Функция по предобработке текстов и сохранению корпуса текстов"""

    # Преобразование списка в датафрейм
    df = pd.DataFrame(corpus)

    # Нормализация текста и удаление слишком коротких текстов
    df['text'] = (df['text'].apply(normalize_text))
    df = df[df['text'].str.len() > 1000]

    return df



def corpus_chunking(corpus, chunk_size=1000, chunk_overlap=200):
    """Функция по разбиению текстов на фрагменты"""

    # Инициализация паттерна разбиения текстов
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Список для записи
    records = []

    for _, row in corpus.iterrows():
        # Разбиение текста на фрагменты
        chunks = splitter.split_text(row['text'])

        # Запись фрагментов с их порядковыми номерами
        for chunk_id, chunk in enumerate(chunks):
            records.append({'source': row['source'], 'author': row['author'],
                            'title': row['title'], 'chunk_id': chunk_id, 'text': chunk
                            })

    # Создание директории для хранения данных
    os.makedirs('data/processed', exist_ok=True)

    # Сохранение корпуса текстов в формате parquet
    corpus = pd.DataFrame(records)
    corpus.to_parquet('data/processed/corpus.parquet', index=False)

    return corpus