import pandas as pd
from src.data_collection.build_corpus import corpus_add_gutenberg, corpus_add_sep, prepare_corpus, corpus_chunking
from src.rag.build_indexer import create_embeddings, build_faiss

# Загрузка текстов произведений
catalog = pd.read_csv('data/raw/gutenberg/philosophy_catalog.csv')

records = []

# Очистка текстов и добавление в общий список
for _, row in catalog.iterrows():
    try:
        corpus_add_gutenberg(records, row)
    except Exception as e:
        print(f"Failed to process {row.title}: {e}")

# Добавление статей SEP в общий список
corpus_add_sep(records)

# Формирование корпуса из текстов и их метаданных
corpus = prepare_corpus(records)

# Разбиение текстов на проиндексированные фрагменты
chunk_corpus = corpus_chunking(corpus)

# Преобразование текстовых фрагментов в эмбеддинги
embeddings_collection = create_embeddings(chunk_corpus)

# Создание объекта индекса для быстрого поиска сходств
build_faiss(embeddings_collection)