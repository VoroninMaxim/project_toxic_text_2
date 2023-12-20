
###!pip install transformers sentencepiece --quiet
###-https://huggingface.co/cointegrated/rubert-tiny-toxicity

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

st.title('Новая улучшенная классификации изображений в облаке Streamlit')

model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba

url = st.text_input('Enter text')

st.write("{:.6f}".format(text2toxicity(url)), "--non-toxic--")