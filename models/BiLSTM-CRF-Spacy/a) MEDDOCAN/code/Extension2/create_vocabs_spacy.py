import os
import pickle
import numpy as np

import spacy

UNK = "${UNK}$".replace("${", "$")  # keep literal $UNK$
NUM = "${NUM}$".replace("${", "$")  # keep literal $NUM$


def main():
    """Create vocab_embeddings_spacy.npz from words.pickle using spaCy token vectors."""
    # Directorio base: carpeta Extension2
    base_dir = os.path.dirname(os.path.abspath(__file__))
    words_path = os.path.join(base_dir, "words.pickle")
    output_path = os.path.join(base_dir, "vocab_embeddings_spacy.npz")

    # 1. Cargar vocabulario de palabras
    with open(words_path, "rb") as f:
        vocab_words = pickle.load(f)

    # vocab_words es un iterable; lo convertimos a lista para fijar el orden
    vocab_list = list(vocab_words)
    word2idx = {w: i for i, w in enumerate(vocab_list)}

    print(f"Vocab size: {len(vocab_list)} palabras")

    # 2. Cargar modelo spaCy con vectores (ajustar si usas otro modelo)
    #    es_core_news_md tiene vectores; si solo tienes es_core_news_sm,
    #    los vectores pueden ser pobres o cero.
    print("Cargando modelo spaCy (es_core_news_md)...")
    try:
        nlp = spacy.load("es_core_news_md")
    except OSError:
        # Fallback a es_core_news_sm si md no está disponible
        print("Aviso: es_core_news_md no encontrado, usando es_core_news_sm. "
              "Los vectores pueden no ser de buena calidad.")
        nlp = spacy.load("es_core_news_sm")

    # 3. Obtener dimensión de los vectores
    test_word = None
    for w in vocab_list:
        if w not in {UNK, NUM}:
            test_word = w
            break
    if test_word is None:
        raise ValueError("No se encontró ninguna palabra normal en el vocabulario.")

    doc = nlp(test_word)
    if len(doc) == 0:
        raise ValueError("spaCy no generó ningún token para la palabra de prueba.")
    emb_dim = doc[0].vector.shape[0]
    print(f"Dimensión de embedding spaCy: {emb_dim}")

    # 4. Matriz de embeddings [vocab_size, emb_dim]
    vocab_size = len(vocab_list)
    emb_matrix = np.zeros((vocab_size, emb_dim), dtype="float32")

    unk_idx = word2idx.get(UNK, None)
    num_idx = word2idx.get(NUM, None)

    # 5. Rellenar la matriz con vectores spaCy
    print("Generando embeddings spaCy para cada palabra del vocabulario...")
    for i, word in enumerate(vocab_list):
        if i % 1000 == 0:
            print(f"  Procesadas {i}/{vocab_size} palabras...")

        if word in {UNK, NUM}:
            continue

        # Procesar la palabra con spaCy y tomar el vector del primer token
        doc = nlp(word)
        if len(doc) == 0:
            # Si por alguna razón no hay tokens, dejamos el vector en cero
            continue
        vec = doc[0].vector.astype("float32")
        emb_matrix[i] = vec

    # 6. Asignar vector medio para UNK y NUM
    valid_rows = np.ones(vocab_size, dtype=bool)
    if unk_idx is not None:
        valid_rows[unk_idx] = False
    if num_idx is not None:
        valid_rows[num_idx] = False

    if valid_rows.any():
        mean_vector = emb_matrix[valid_rows].mean(axis=0)
    else:
        mean_vector = np.zeros(emb_dim, dtype="float32")

    if unk_idx is not None:
        emb_matrix[unk_idx] = mean_vector
    if num_idx is not None:
        emb_matrix[num_idx] = mean_vector

    # 7. Guardar en .npz con clave "embeddings"
    print(f"Guardando matriz de embeddings en: {output_path}")
    np.savez_compressed(output_path, embeddings=emb_matrix)

    print("Listo. Ahora puedes usar vocab_embeddings_spacy.npz en tu modelo.")


if __name__ == "__main__":
    main()
