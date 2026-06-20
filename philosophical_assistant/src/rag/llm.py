import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ollama import chat

# Инициализация модели, преобразующей тексты в эмбеддинги, корпуса текстов и объекта индекса
model = SentenceTransformer('intfloat/multilingual-e5-base')
chunk_df = pd.read_parquet('data/processed/corpus.parquet')
faiss_index = faiss.read_index('data/processed/philosophy.index')



def query_search(query, k=10):
    """Функция, производящая быстрый поиск эмбеддингов, наиболее похожих на запрос"""

    # Преобразование запроса в эмбеддинг
    query_embedding = model.encode([f'query: {query}'], normalize_embeddings=True).astype(np.float32)

    # Поиск наиболее близких эмбеддингов
    distances, ids = faiss_index.search(query_embedding, k=k)

    return distances, ids



def generate_answer(question, context):
    """Функция, передающая модели промпт с контекстом и запросом
                    и возвращающая ответ модели"""

    # Промпт, передаваемый модели при каждом запросе
    prompt = f"""   Ты философский ассистент.
                    Отвечай на вопросы, используя только получаемый контекст.
                    Принимая вопросы, определяй язык, на котором они написаны, и отвечай на этом же языке.
                    Не используй собственные знания. Не додумывай факты.
                    Если информации недостаточно, ответь: 
                    "В предоставленном контексте нет достаточной информации для ответа на этот вопрос."

                    Контекст:
                    {context}

                    Вопрос:
                    {question}

                    Ответ:
                    """

    # Получение ответа модели
    response = chat(model="qwen3:8b", messages=[{"role": "user", "content": prompt}], think=False)

    return response.message.content


def ask(question):
    """Функция, генерирующая контекст по запросу, передающая их модели
                    и возвращающая ответ модели"""

    # Получение индексов текстов, входящих в контекст
    distances, ids = query_search(question, k=5)

    # Получение текстов по индексам
    chunks = chunk_df.iloc[ids[0].tolist()]

    context_parts = []

    # Формирование списка из текстов и их источников
    for _, row in chunks.iterrows():
        context_parts.append(f"""   Author: {row['author']}
                                    Title: {row['title']}

                                    {row['text']}
                                    """)

    # Формирование общего контекста
    context = "\n\n".join(context_parts)

    # Формирование списков источников
    sources = chunks[["author", "title"]].drop_duplicates().to_dict("records")

    # Возвращение ответа модели и источников
    return generate_answer(question, context), sources