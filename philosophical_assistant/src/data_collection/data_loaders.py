import os
import time
import requests
import re
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset



def search_author(author):
    """Функция, выполняющая поиск произведений по автору,
                    получая их метаданные"""

    # Инициализация адреса и параметров поиска
    url = 'https://gutendex.com/books'
    params = {'search': author}

    # Список для сохранения результатов
    all_results = []

    # Цикл для прохода по всем страницам ответов
    while url:

        # поиск по автору
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Сохранение данных в общем списке
        data = response.json()
        all_results.extend(data['results'])

        # Переход на следующую страницу и удаление параметров
        url = data['next']
        params = None

    return all_results



def sanitize_filename(filename: str) -> str:
    """Функция, очищающая названия файлов от символов,
        из-за которых возникнут проблемы при чтении"""

    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', ' ', filename)

    return filename.strip()



def parse_books(target_author, language='en'):
    """Функция по парсингу данных из метаданных
                произведений для автора"""

    # Получение метаданных
    data = search_author(target_author)

    books = []

    # Цикл получения метаданных по произведению
    for book in data:

        # Фильтрация по языку от одних и тех же книг на разных языках
        if language not in book['languages']:
            continue

        # Получение информации об авторах
        authors = [author['name'] for author in book['authors']]

        # Выделение фамилии автора для фильтрации
        surname = target_author.split()[-1].lower()

        # Фильтрация по автору
        if not any( surname in author.lower() for author in authors):
            continue

        # Получение данных о категориях
        categories = [re.sub(r'^.*?: ', '', category) for category in book['bookshelves']]

        # Фильтрация по категории
        if 'Philosophy & Ethics' not in categories:
            continue

        # Получение ссылки на текст книги
        txt_url = ( book['formats'].get('text/plain; charset=utf-8')
                    or book['formats'].get('text/plain')
                    )

        # Добавление метаданных в список
        books.append({'id': book['id'], 'author': ', '.join(authors),
                      'title': book['title'], 'categories': categories,
                      'downloads': book['download_count'],
                      'source': 'Project Gutenberg', 'url': txt_url,
                      'local_path': f'data/raw/gutenberg/{sanitize_filename(book['title'])}.txt'
                      })

    return books



def load_gutenberg_data(philosophers_list):
    """Функция, загружающая метаданные произведений по заданным авторам"""

    all_books = []

    # Получение метаданных произведений по автору с сохранением в общий список
    for philosopher in tqdm(philosophers_list):
        all_books.extend(
            parse_books(philosopher)
        )
        time.sleep(1)

    # Создание директории для хранения данных о произведениях
    os.makedirs('data/raw/gutenberg', exist_ok=True)

    # Преобразование данных в датафрейм, удаление записей с отсутствующей
    # ссылкой на текст, дубликатов и сохранение в csv
    df = pd.DataFrame(all_books)
    df = (df.dropna(subset=['url']).
          drop_duplicates(subset=['author', 'title']).
          sort_values('downloads', ascending=False)
          )
    df.to_csv('data/raw/gutenberg/philosophy_catalog.csv', index=False)
    return df



def download_book(name, url):
    """Функция, загружающая тексты произведений по url"""

    # Получение текста произведения
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Очистка названия текста
    safe_name = sanitize_filename(name)

    # Запись текста
    with open(f'data/raw/gutenberg/{safe_name}.txt', 'w', encoding='utf-8') as f:
        f.write(response.text)

    # Отслеживание процесса
    print(f'Downloaded: {name}')



def load_sep():
    """Функция, загружающая датасет SEP"""

    # Создание директории для хранения данных
    os.makedirs('data/raw/sep', exist_ok=True)

    # Загрузка датафрейма
    dataset = load_dataset('AiresPucrs/stanford-encyclopedia-philosophy')

    # Преобразование в датафрейм
    df = dataset['train'].to_pandas()

    # Объединение уже разбитых на фрагменты тестов
    sep_merged = (  df.groupby('metadata', as_index=False)
                    .agg({'text': ' '.join,'category': 'first'
                    }))

    # Преобразование в формат parquet
    sep_merged.to_parquet('data/raw/sep/sep.parquet')