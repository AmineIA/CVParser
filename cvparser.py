import streamlit as st
import spacy
import json
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np

# Chargement du modèle NLP
nlp = spacy.load('en_core_web_sm')

# Fonction pour extraire du texte à partir de différents formats de fichiers
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        text = file.read().decode("utf-8")
    return text

# Fonction pour extraire les compétences du texte
def extract_skills(text):
    doc = nlp(text)
    skills = []
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "GPE", "PRODUCT"]:  # Exemples de labels qui peuvent correspondre aux compétences
            skills.append(ent.text)
    return skills

# Fonction pour générer les embeddings des mots-clés
def generate_embeddings(keywords, model, tokenizer):
    inputs = tokenizer(keywords, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Fonction pour calculer la similarité cosinus
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Chargement du modèle et du tokenizer pour les embeddings
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Interface Streamlit
st.title("CV Catcher: Analyse de CV et Correspondance avec Offres")
uploaded_cv = st.file_uploader("Choisissez un fichier CV", type=["pdf", "docx", "txt"])

if uploaded_cv is not None:
    raw_text_cv = extract_text(uploaded_cv)
    if raw_text_cv:
        st.text_area("Texte brut extrait du CV", raw_text_cv, height=200)
        skills_cv = extract_skills(raw_text_cv)
        st.subheader("Compétences extraites du CV")
        st.write(skills_cv)

        offer_text = st.text_area("Collez le texte de l'offre d'emploi ou de stage ici")
        if offer_text:
            skills_offer = extract_skills(offer_text)
            st.subheader("Compétences extraites de l'offre")
            st.write(skills_offer)

            # Générer les embeddings
            embeddings_cv = generate_embeddings(skills_cv, model, tokenizer)
            embeddings_offer = generate_embeddings(skills_offer, model, tokenizer)

            # Calculer la similarité moyenne
            similarities = [cosine_similarity(cv, offer) for cv in embeddings_cv for offer in embeddings_offer]
            similarity_score = np.mean(similarities) if similarities else 0

            st.subheader("Score de similarité")
            st.write(f"Le score de similarité entre le CV et l'offre est de : {similarity_score:.2f}")

        st.download_button(
            label="Télécharger Texte Brut du CV",
            data=raw_text_cv.encode('utf-8'),
            file_name="cv_text.txt",
            mime="text/plain"
        )
    else:
        st.error("Impossible d'extraire le texte du fichier. Veuillez essayer un autre fichier.")
