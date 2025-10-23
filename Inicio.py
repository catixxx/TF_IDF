import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# --- 🌸 Estilos tipo Pinterest femenino ---
st.markdown("""
    <style>
    body {
        background-color: #fff8fa;
    }
    .stApp {
        background-color: #fff8fa;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3 {
        color: #c86b9b !important;
        text-align: center;
        font-weight: 600;
    }
    .stTextArea, .stTextInput {
        background-color: #ffffff !important;
        border: 1.5px solid #f2b3d1 !important;
        border-radius: 15px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #f7b7c4, #f1a7de);
        color: white;
        border-radius: 25px;
        padding: 0.6em 1.2em;
        font-size: 1em;
        border: none;
        font-weight: 600;
        box-shadow: 0px 4px 10px rgba(249, 150, 200, 0.4);
        transition: all 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #fbb0ce, #e985b5);
        transform: scale(1.05);
    }
    .stDataFrame {
        border-radius: 12px !important;
        background-color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 🌷 Encabezado ---
st.markdown("<h1>✨ Demo de TF-IDF con Preguntas y Respuestas 💕</h1>", unsafe_allow_html=True)
st.write("""
🩷Cada línea se trata como un documento (puede ser una frase, un párrafo o un texto más largo).
Los documentos y las preguntas deben estar en inglés, ya que el análisis está configurado para ese
idioma.
La aplicación aplica normalización y stemming para que palabras como playing y play se consideren equivalentes.
""")

# --- 💌 Entrada de texto y pregunta ---
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area(
        "📜 Escribe tus documentos (uno por línea, en inglés):",
        "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
        height=180
    )

with col2:
    question = st.text_input("💭 Escribe tu pregunta (en inglés):", "Who is playing?")

# --- 🌼 Procesamiento del texto ---
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# --- 🌈 Botón principal ---
if st.button("🌸 Analizar textos"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )
        X = vectorizer.fit_transform(documents)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.markdown("### 🌷 Matriz TF-IDF")
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        # --- 🌹 Mostrar resultados ---
        st.markdown("### 💖 Resultado del análisis")
        st.success(f"**Tu pregunta:** {question}")
        st.info(f"**Respuesta más relacionada:** {best_doc}")
        st.markdown(f"✨ **Similitud:** `{best_score:.3f}`")

        # --- 🕊️ Tabla de similitudes ---
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.markdown("### 🩰 Similitudes entre documentos")
        st.dataframe(sim_df.sort_values("Similitud", ascending=False), use_container_width=True)

        # --- 🌺 Palabras coincidentes ---
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
        st.markdown("### 🌼 Stems de la pregunta presentes en el documento elegido:")
        st.write(", ".join(matched) if matched else "No se encontraron coincidencias 💭")
