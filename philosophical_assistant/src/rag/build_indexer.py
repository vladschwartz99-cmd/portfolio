from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# Выбор устройства для расчетов
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Инициализация модели, преобразующей тексты в эмбеддинги
model = SentenceTransformer('intfloat/multilingual-e5-base', device=device)



def create_embeddings(df):
    """Функция, преобразующая тексты в эмбеддинги"""

    # Получение списка из текстов в нужном формате
    passages = [f'passage: {text}' for text in df['text'].tolist()]

    # Преобразование тестов
    embeddings = model.encode(passages, batch_size=16, show_progress_bar=True)

    # Нормализация эмбеддингов
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    # Сохранение для последующего быстрого доступа
    np.save('data/processed/embeddings.npy', embeddings)

    return embeddings



def build_faiss(embeddings):
    """Функция, создающая объект индекса на основе эмбеддингов"""

    # Создание объекта индекса
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])

    # Заполнение эмбеддингами
    faiss_index.add(embeddings)

    # Сохранение для последующего быстрого доступа
    faiss.write_index(faiss_index, 'data/processed/philosophy.index')