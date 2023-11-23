import torch
import transformers
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

device="cpu"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

model=BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=3)

model.load_state_dict(torch.load("model_state_dict.pth",map_location=torch.device("cpu")))

model.eval()

def predict_sentiment(text):
    token=tokenizer.encode_plus(text,
                                add_special_tokens = True,
                                max_length = 128,
                                pad_to_max_length = True,
                                return_attention_mask = True,
                               return_tensors="pt")

    ids=token["input_ids"].to(device)
    mask=token["attention_mask"].to(device)

    with torch.no_grad():
        pred=model(ids,mask)
        logits=pred.logits

    predicted_label = torch.argmax(logits, dim=1).item()

    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

    predicted_sentiment = sentiment_mapping[predicted_label]

    return predicted_sentiment