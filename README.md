# 🐾 Cat Health Assistant (LoRA GPT-2 + Symptom Predictor)
A hybrid app that predicts possible illnesses from breed + symptom similarity, and answers free-form cat-health questions with a GPT-2 model fine-tuned via LoRA.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)


# How we built it
- Dataset – 1 k Q-A pairs (CSV) · cleaned & hosted on 🤗 Datasets
- Fine-tuning – LoRA (r = 16) on GPT-2 small, 3 epochs, single A10 GPU
- Space – Streamlit UI served directly from the Hub

  



