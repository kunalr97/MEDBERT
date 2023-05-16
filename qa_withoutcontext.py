import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

token = "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"
tokenizer = AutoTokenizer.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)
model = AutoModelForQuestionAnswering.from_pretrained("GerMedBERT/medbert-512",use_auth_token=token)

# Define your question
question = "Was sind die Symptome einer Lungenentzündung?"


# Prepare the input context and question for the model
# by encoding them into IDs, which is then
# converted to a tensor that the model expects
input_ids = tokenizer.encode_plus(question, add_special_tokens=True, return_tensors='pt')

# Use the model to get the answer to the question
outputs = model(**input_ids)
start, end = outputs.start_logits, outputs.end_logits
answer_start = torch.argmax(start)
answer_end = torch.argmax(end) + 1

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids["input_ids"][0][answer_start:answer_end]))

print(f"Question: {question}")
print(f"Answer: {answer}")