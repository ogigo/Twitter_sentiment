import torch
from dataloaders import train_dataloader,valid_dataloader
from model import bert_model
from tqdm import tqdm
from optimize import optimizer,criterion
from sklearn.metrics import accuracy_score




num_epochs=5
device="cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(num_epochs):
    bert_model.train()
    total_loss=0.0
    for batch in tqdm(train_dataloader,desc=f"Epoch {epoch + 1}/{num_epochs}"):
        ids=batch["ids"].to(device)
        mask=batch["mask"].to(device)
        target=batch["labels"].to(device)

        output=bert_model(input_ids=ids,
                            attention_mask=mask,
                            labels=target)

        optimizer.zero_grad()
        loss=output.loss
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    print(f"Training Loss: {average_loss}")

    # Evaluation loop
    bert_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            input_ids = batch['ids'].to(device)
            attention_mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = bert_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {accuracy}")