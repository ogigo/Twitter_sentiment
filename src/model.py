import torch
import transformers


device="cuda" if torch.cuda.is_available() else "cpu"

bert_model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=3)

bert_model.to(device)