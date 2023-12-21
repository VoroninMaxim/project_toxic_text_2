
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

st.write("""
    # Мое первое веб приложение выложенное в Streamlit

    Это коинтегрированная модель Руберта-Тайни, Предназначенная для классификации токсичности 
    и неуместности коротких неформальных русских текстов.

""")

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
#----------------------------------------------------------------

if 'my_lst' not in st.session_state:
    st.session_state['my_lst'] = []

with st.expander("Example"):
    user_input = st.text_input("Enter a key")
    add_button = st.button("Add", key='add_button')
    if add_button:
        if len(user_input) > 0:
            st.session_state['my_lst'] += [(text2toxicity(user_input))]
            st.write( st.session_state['my_lst'] )
        else:
            st.warning("Enter text")

