import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import streamlit as st

@st.cache_resource
def load_model():
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)
    model = AutoModelForMaskedLM.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)
    return tokenizer, model

# token = "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"
# tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)
# model = AutoModelForMaskedLM.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)

tokenizer, model = load_model()

# text = "Die [MASK] ist ein Organ des Menschen."
# text ='''Der 56-jährige Patient klagt über anhaltende Schmerzen in der Brust und Atemnot. 
# Eine körperliche Untersuchung ergab eine verminderte Sauerstoffsättigung und ein Röntgenbild zeigte Anzeichen einer [MASK]. 
# Der Patient wurde umgehend in die Notaufnahme eingewiesen und erhält nun eine Sauerstofftherapie sowie eine Antibiotikabehandlung '''
st.title("German MedBERT Masked Language Model")
st.write("You can use the following example text:  \n  \n", "Der 56-jährige Patient klagt über anhaltende Schmerzen in der Brust und Atemnot.Eine körperliche Untersuchung ergab eine verminderte Sauerstoffsättigung und ein Röntgenbild zeigte Anzeichen einer [MASK].Der Patient wurde umgehend in die Notaufnahme eingewiesen und erhält nun eine Sauerstofftherapie sowie eine Antibiotikabehandlung")
text = st.text_area("Enter a German clinical text with a [MASK] token:",height=20, max_chars=500)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write(' ')
with col2:
    st.write(' ')
with col3:
    # Create a enter button
    button = st.button("Predict")
with col4:
    st.write(' ')
# Check if the user has entered a text with a [MASK] token
if text.count("[MASK]") == 1 and button:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    token_logits = model(input_ids)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    st.write("---------------------Predictions---------------------")
    for token in top_5_tokens:
        st.write(tokenizer.decode([token]))
else:
    st.write("Please enter a text with a [MASK] token!")
st.markdown("> MEDBERT.de: A Comprehensive German BERT Model for the Medical Domain  \n Keno K. Bressem and Jens-Michalis Papaioannou and Paul Grundmann. 2023.  \n arXiv preprint arXiv:2303.08179 https://doi.org/10.48550/arXiv.2303.08179")
st.write(' ')
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image('IAIS.PNG', width=200)
with col3:
    st.write(' ')
st.write("<p style='text-align: center;'>Created by Dario Antweiler and Kunal Runwal", unsafe_allow_html=True)


