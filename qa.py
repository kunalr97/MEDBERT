import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

token = "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"
tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)
model = AutoModelForQuestionAnswering.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)

# Define your question and context
question = "Welche Anzeichen zeigte das Röntgenbild?"
context = '''Der 56-jährige Patient klagt über anhaltende Schmerzen in der Brust und Atemnot. Eine körperliche Untersuchung ergab eine verminderte Sauerstoffsättigung und ein Röntgenbild zeigte Anzeichen einer Lungenentzündung.
 Der Patient wurde umgehend in die Notaufnahme eingewiesen und erhält nun eine Sauerstofftherapie sowie eine Antibiotikabehandlung'''

# Prepare the input context and question for the model
# by encoding them into IDs, which is then
# converted to a tensor that the model expects
encoding = tokenizer.encode_plus(question,context,add_special_tokens=True,return_tensors='pt')

# Use the model to get the answer to the question
input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores) + 1
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))

print("Antwort:", answer)