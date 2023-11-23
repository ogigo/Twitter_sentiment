import torch
from dataset import train_df,valid_df
from torch.utils.data import Dataset,DataLoader
from inference import tokenizer


class Twitter(Dataset):
    def __init__(self,df):
        self.df=df
        self.text=self.df.text.values
        self.label=self.df.label.values

    def __len__(self):
        return len(self.text)

    def __getitem__(self,index):
        text=self.text[index]

        token=tokenizer.encode_plus(text,
                                   add_special_tokens = True,
                                   max_length = 128,
                                   pad_to_max_length = True,
                                   return_attention_mask = True)

        input_ids=torch.tensor(token["input_ids"],dtype=torch.long)
        mask=torch.tensor(token["attention_mask"],dtype=torch.long)
        token_type_ids=torch.tensor(token["token_type_ids"],dtype=torch.long)
        target=torch.tensor(self.label[index],dtype=torch.long)

        return {
            "ids": input_ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "labels": target
        }
    
train_dataset=Twitter(train_df)
valid_dataset=Twitter(valid_df)

train_dataloader=DataLoader(train_dataset,batch_size=32)
valid_dataloader=DataLoader(valid_dataset,batch_size=32)