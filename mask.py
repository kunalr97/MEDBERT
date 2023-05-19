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
text = st.text_area("Enter a German clinical text with a [MASK] token:",height=20, max_chars=500)
if text:
    input_ids = tokenizer.encode(text, return_tensors="pt")
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    token_logits = model(input_ids)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(tokenizer.decode([token]))
        st.write("Predicted [MASK] Token: ",tokenizer.decode([token]))
