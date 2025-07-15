import streamlit as st
import spacy
import stanza
import transformers
from transformers import logging    
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Chargement des modèles
@st.cache_resource
def load_spacy():
    return spacy.load("fr_core_news_md")

@st.cache_resource
def load_stanza():
    stanza.download("fr")
    return stanza.Pipeline(lang="fr", processors="tokenize,ner")

@st.cache_resource
def load_camembert():
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Interface utilisateur
st.title("📌 Démonstration de NER (Reconnaissance d’Entités Nommées)")
st.write("Saisissez une phrase en français pour détecter les entités nommées avec le modèle de votre choix.")

text = st.text_area("✍️ Entrez votre phrase ici :", "Emmanuel Macron est le président de la République française.")

model_choice = st.selectbox("🧠 Choisir le modèle NER :", ["spaCy", "Stanza", "CamemBERT (HuggingFace)"])

if st.button("🔍 Analyser"):
    if model_choice == "spaCy":
        nlp = load_spacy()
        doc = nlp(text)
        st.markdown("### 📄 Résultat spaCy")
        for ent in doc.ents:
            st.write(f"{ent.text} ➜ {ent.label_}")
        st.code(" ".join([f"{token.text} -> {token.ent_iob_}-{token.ent_type_ if token.ent_type_ else 'O'}" for token in doc]))

    elif model_choice == "Stanza":
        nlp = load_stanza()
        doc = nlp(text)
        st.markdown("### 📄 Résultat Stanza")
        for sentence in doc.sentences:
            for ent in sentence.ents:
                st.write(f"{ent.text} ➜ {ent.type}")
            bio_tags = [f"{word.text} -> {word.ner}" for word in sentence.words]
            st.code("\n".join(bio_tags))

    elif model_choice == "CamemBERT (HuggingFace)":
        ner_pipe = load_camembert()
        results = ner_pipe(text)
        st.markdown("### 📄 Résultat CamemBERT")
        for ent in results:
            st.write(f"{ent['word']} ➜ {ent['entity_group']} ({ent['score']:.2f})")
        st.code("\n".join([f"{ent['word']} -> B-{ent['entity_group']}" for ent in results]))

