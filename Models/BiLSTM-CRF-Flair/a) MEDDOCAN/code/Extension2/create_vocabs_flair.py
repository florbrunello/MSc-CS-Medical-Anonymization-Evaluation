import os
import pickle
import numpy as np

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

UNK = "$UNK$"
NUM = "$NUM$"

def main():
    # Rutas relativas desde create_vocabs_flair.py
    base_dir = os.path.dirname(os.path.abspath(__file__))  # .../Extension2
    words_path = os.path.join(base_dir, "words.pickle")
    output_path = os.path.join(base_dir, "vocab_embeddings_flair.npz")

    # 1. Cargar vocabulario de palabras
    with open(words_path, "rb") as f:
        vocab_words = pickle.load(f)

    # vocab_words es algún iterable (lista, set, etc.)
    # Lo convertimos a lista fijando un orden consistente
    vocab_list = list(vocab_words)

    # Guardamos también un mapa palabra->índice por si lo quieres usar
    word2idx = {w: i for i, w in enumerate(vocab_list)}

    print(f"Vocab size: {len(vocab_list)} palabras")

    # 2. Definir embeddings Flair
    print("Cargando embeddings Flair (es-forward y es-backward)...")
    embedding_types = [
        FlairEmbeddings('es-forward'),
        FlairEmbeddings('es-backward'),
    ]
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 3. Obtener dimensión de los vectores (emb_dim)
    #    Hacemos una prueba con una palabra cualquiera (por ejemplo la primera que no sea UNK/NUM)
    test_word = None
    for w in vocab_list:
        if w not in {UNK, NUM}:
            test_word = w
            break
    if test_word is None:
        raise ValueError("No se encontró ninguna palabra normal en el vocabulario.")

    test_sentence = Sentence([test_word])
    embeddings.embed(test_sentence)
    emb_dim = len(test_sentence[0].embedding)
    print(f"Dimensión de embedding Flair: {emb_dim}")

    # 4. Crear matriz de embeddings [vocab_size, emb_dim]
    vocab_size = len(vocab_list)
    emb_matrix = np.zeros((vocab_size, emb_dim), dtype="float32")

    # Para UNK y NUM, podemos rellenar luego. De momento, guardamos sus índices.
    unk_idx = word2idx.get(UNK, None)
    num_idx = word2idx.get(NUM, None)

    # 5. Rellenar la matriz
    print("Generando embeddings Flair para cada palabra del vocabulario...")
    for i, word in enumerate(vocab_list):
        # Opcional: mensajes de progreso
        if i % 1000 == 0:
            print(f"  Procesadas {i}/{vocab_size} palabras...")

        if word in {UNK, NUM}:
            # Lo rellenaremos después con algo específico
            continue

        # Crear oración con la palabra
        sentence = Sentence([word])
        embeddings.embed(sentence)

        # Obtener vector del único token
        vec = sentence[0].embedding.cpu().detach().numpy().astype("float32")
        emb_matrix[i] = vec

    # 6. Rellenar UNK y NUM si existen
    #    Estrategia: vector medio del vocabulario (simple, pero razonable)
    valid_rows = np.ones(vocab_size, dtype=bool)
    if unk_idx is not None:
        valid_rows[unk_idx] = False
    if num_idx is not None:
        valid_rows[num_idx] = False

    mean_vector = emb_matrix[valid_rows].mean(axis=0) if valid_rows.any() else np.zeros(emb_dim, dtype="float32")

    if unk_idx is not None:
        emb_matrix[unk_idx] = mean_vector
    if num_idx is not None:
        emb_matrix[num_idx] = mean_vector

    # 7. Guardar en .npz con la clave "embeddings" (como espera get_fasttext_vectors)
    print(f"Guardando matriz de embeddings en: {output_path}")
    np.savez_compressed(output_path, embeddings=emb_matrix)

    print("Listo. Ahora puedes usar vocab_embeddings_flair.npz en tu modelo.")

if __name__ == "__main__":
    main()