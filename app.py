import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import stanza

# ---------------------------
# Configuration visuelle
# ---------------------------
st.set_page_config(
    page_title="NER Presse 🇫🇷",
    page_icon="📍",
    layout="centered"
)

st.markdown("""
<style>
.big-title {
    font-size: 36px;
    font-weight: bold;
    color: #1f77b4;
}
.entity-badge {
    display: inline-block;
    background-color: #f0f0f0;
    color: #333;
    border-radius: 0.5rem;
    padding: 0.4rem 0.7rem;
    margin: 0.2rem;
    font-size: 16px;
    font-weight: 500;
    border-left: 5px solid #00cc99;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📍 NER - Extraction d’entités nommées dans la presse</div>', unsafe_allow_html=True)

st.markdown("""
Bienvenue dans cette application interactive de reconnaissance d'entités nommées (**NER**) sur des textes en **français**.

🧠 Modèles disponibles :
- 🤗 CamemBERT (HuggingFace)
- 🧪 spaCy (`fr_core_news_md`)
- 🔎 Stanza (`fr`)

---

""", unsafe_allow_html=True)

# ---------------------------
# Chargement des modèles
# ---------------------------

@st.cache_resource
def load_camembert_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@st.cache_resource
def load_spacy_model():
    return spacy.load("fr_core_news_md")

@st.cache_resource
def load_stanza_model():
    stanza.download("fr")
    return stanza.Pipeline(lang="fr", processors="tokenize,ner")

camembert_pipeline = load_camembert_pipeline()
spacy_model = load_spacy_model()
stanza_model = load_stanza_model()

# ---------------------------
# Interface utilisateur
# ---------------------------

with st.expander("💬 Exemple de texte", expanded=False):
    st.markdown("""
    ```
    Emmanuel Macron s'est rendu à Bruxelles pour une réunion de l'Union Européenne.
    Le président de TotalEnergies a rencontré les dirigeants de l’ONU à Genève.
    ```
    """)

text_input = st.text_area("✏️ Entrez un texte en français", height=200)

model_choice = st.selectbox("🧠 Choisissez un modèle NER", [
    "CamemBERT (HuggingFace)",
    "spaCy",
    "Stanza"
])

if st.button("🚀 Extraire les entités"):
    st.markdown("---")
    st.subheader("📌 Résultats :")

    if not text_input.strip():
        st.warning("Veuillez saisir un texte pour extraire les entités.")
    else:
        with st.spinner("Analyse en cours..."):
            if model_choice == "CamemBERT (HuggingFace)":
                results = camembert_pipeline(text_input)
                entities = [(ent['word'], ent['entity_group']) for ent in results]

            elif model_choice == "spaCy":
                doc = spacy_model(text_input)
                entities = [(ent.text, ent.label_) for ent in doc.ents]

            elif model_choice == "Stanza":
                doc = stanza_model(text_input)
                entities = [(ent.text, ent.type) for sent in doc.sentences for ent in sent.ents]

            else:
                entities = []

        if entities:
            st.success(f"✅ {len(entities)} entité(s) détectée(s) :")
            for ent, label in entities:
                color = "#FF6961" if label == "PER" else "#77DD77" if label == "LOC" else "#779ECB"
                st.markdown(f'<span class="entity-badge">🟢 <strong>{ent}</strong> — <em>{label}</em></span>', unsafe_allow_html=True)
        else:
            st.info("Aucune entité détectée dans le texte.")

# ---------------------------
# Pied de page
# ---------------------------

st.markdown("---")
st.markdown("🧪 *Projet NER réalisé dans le cadre du Master IA - NLP*")
st.markdown("📄 *Source des modèles : [HuggingFace](https://huggingface.co/), [spaCy](https://spacy.io/), [Stanza](https://stanfordnlp.github.io/stanza/)*" )