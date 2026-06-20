import pandas as pd
from src.data_collection.data_loaders import load_gutenberg_data, download_book, load_sep

# Список интересных философов
PHILOSOPHERS = ['Plato', 'Aristotle', 'Epictetus', 'Marcus Aurelius', 'Augustine',
                'Machiavelli', 'Thomas Aquinas', 'Descartes', 'Spinoza', 'Hobbes',
                'Leibniz', 'Locke', 'Rousseau', 'Hume', 'Kant', 'Hegel', 'Mill',
                'Schopenhauer', 'Nietzsche', 'Bergson', 'Dewey'
                ]


# Загрузка списка произведений этих философов
load_gutenberg_data(PHILOSOPHERS)

# Загрузка текстов произведений
catalog = pd.read_csv('data/raw/gutenberg/philosophy_catalog.csv')
for _, row in catalog.iterrows():
    try:
        download_book(row['title'], row['url'])
    except Exception as e:
        print(f"Failed: {row['title']} ({e})")

# Загрузка датафрейма SEP
load_sep()