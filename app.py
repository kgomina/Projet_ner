import streamlit as st
import spacy
import stanza
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import classification_report

@st.cache_resource
def load_models():
    # spaCy
    nlp_spacy = spacy.load("fr_core_news_md")

    # Stanza
    stanza.download("fr")
    nlp_stanza = stanza.Pipeline("fr")

    # CamemBERT (HuggingFace)
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    hf_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    return nlp_spacy, nlp_stanza, hf_pipeline

nlp_spacy, nlp_stanza, hf_pipeline = load_models()

st.title("📌 Reconnaissance d'entités nommées (NER) avec modèles NLP français")

text_input = st.text_area("✍️ Entrez une phrase :", "Emmanuel Macron est le président de la République française.")
model_choice = st.selectbox("🧠 Choisissez un modèle NER :", ["spaCy", "Stanza", "CamemBERT (HuggingFace)"])

if st.button("🔍 Analyser"):
    st.markdown("### Résultats")

    if model_choice == "spaCy":
        doc = nlp_spacy(text_input)
        for ent in doc.ents:
            st.markdown(f"- **{ent.text}** → *{ent.label_}*")

    elif model_choice == "Stanza":
        doc = nlp_stanza(text_input)
        for sent in doc.sentences:
            for ent in sent.ents:
                st.markdown(f"- **{ent.text}** → *{ent.type}*")

    else:  # CamemBERT (HuggingFace)
        results = hf_pipeline(text_input)
        for ent in results:
            st.markdown(f"- **{ent['word']}** → *{ent['entity_group']}*")

    st.success("Analyse terminée ✅")


# ---------------------------
# Pied de page
# ---------------------------
st.markdown("---")
st.markdown("🧪 *Projet NER réalisé dans le cadre du Master IA - NLP*")
st.markdown("📄 *Sources : [HuggingFace](https://huggingface.co/), [spaCy](https://spacy.io/), [Stanza](https://stanfordnlp.github.io/stanza/)*")
