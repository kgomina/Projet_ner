# 📍 Application Streamlit NER - Français

Cette application permet de détecter automatiquement les entités nommées dans un texte en français, à l’aide de trois modèles :

- 🤗 **CamemBERT (HuggingFace)**
- 🧠 **spaCy (fr_core_news_md)**
- 🔎 **Stanza (fr)**

## 🛠️ Installation

Crée un environnement virtuel et installe les dépendances :

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
