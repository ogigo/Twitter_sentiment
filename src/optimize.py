import torch
import torch.nn as nn
import torch.optim as optim
from model import bert_model
from transformers import AdamW, get_linear_schedule_with_warmup


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(bert_model.parameters(), lr=2e-5)