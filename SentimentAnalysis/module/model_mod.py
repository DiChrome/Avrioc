# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LstmClassifier(nn.Module):
    def __init__(self, vocab_size, dim_list, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_list[0])

        self.lstm = nn.LSTM(
            input_size=dim_list[0],
            hidden_size=dim_list[1],
            batch_first=True, #[BatchToken]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim_list[1], dim_list[2])

    def forward(self, input_id_tensor):
        logit = self.embedding(input_id_tensor)#[BatchTokenEmbedding]
        output, (hn, cn) = self.lstm(logit)#hn=[LayerBatchHidden]
        h = hn[-1]#[BatchHidden]
        logit = self.dropout(h)
        logit = self.linear(logit)#[BatchHidden]
        return logit

# %%
class TrainModel:
    def __init__(self, params, train_dataloader, val_dataloader, model, loss_func, optimizer):
        self.params = params
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

# %%
    def model_iteration(self, batch):
        input_id_tensor = batch["input_ids"].to(self.params["device"])
        label_tensor = batch["label"].to(self.params["device"])

        logit_tensor = self.model(input_id_tensor)#[BatchHidden]
        logit_tensor = logit_tensor.view(-1)#[Batch]
        loss = self.loss_func(logit_tensor.float(), label_tensor.float())
        return logit_tensor, label_tensor, loss

# %%
    
    def model_epoch(self, mode):
        print(f"model_epoch({mode=})")
        total_loss = 0
        correct_pred = 0
        total_pred = 0
        if mode == "train":
            self.model.train()
            dataloader = self.train_dataloader
        elif mode == "val":
            self.model.eval()
            dataloader = self.val_dataloader

        for batch_num, batch in enumerate(self.train_dataloader):
            print(f"\r{batch_num=}", end="")
            if mode == "train":
                self.optimizer.zero_grad()
            logit_tensor, label_tensor, loss = self.model_iteration(batch)

            total_loss+=loss.item()
            total_pred += label_tensor.size(0)
            pred_tensor = (torch.sigmoid(logit_tensor) > 0.5).int()
            correct_pred += (pred_tensor == label_tensor).sum().item()
            # print(f"{loss=}|{loss.shape=}")
            
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # if batch_num >= 10:  # To do: remove
            #     break
            
        total_pred += label_tensor.size(0)
        loss = total_loss/len(dataloader)
        accuracy = correct_pred/total_pred

        print("")        
        return loss, accuracy

# %%
    def train_model(self):
        print(f"train_model()")
        output_list = []
        for epoch in range(1, self.params["epoch"]+1):
            train_loss, train_accuracy = self.model_epoch("train")
            val_loss, val_accuracy = self.model_epoch("val")
            print(f"\nepoch={epoch}/{self.params['epoch']}|{train_loss=}|{val_loss=}")
            print(f"{train_accuracy=}|{val_accuracy=}")
            output_list.append({"epoch":epoch, "train_loss": train_loss, "val_loss": val_loss,
                                "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
        return output_list

# %%
    def plot_metrics(self, df):
        print("plot_metrics()")
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], marker='x', label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], marker='x', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid(True)

            
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_accuracy'], marker='x', label='Train Accuracy')
        plt.plot(df['epoch'], df['val_accuracy'], marker='x', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        return plt

# %%
class InferModel:
    def __init__(self, tokenizer_params, model):
        self.tokenizer_params = tokenizer_params
        self.model = model
    # %%
    def predict(self, text):
        # tokenizer_params["return_tensors"]
        token_id = self.tokenizer.encode(text, padding=self.tokenizer_params["padding"],
                                    max_length = self.tokenizer_params["max_length"],
                                    truncation=self.tokenizer_params["truncation"])
        input_tensor = torch.tensor(token_id).unsqueeze(0)
        with torch.no_grad():
            logit_tensor = self.model(input_tensor).view(-1)
        
        pred_tensor = (torch.sigmoid(logit_tensor) > 0.5).int()
        pred = pred_tensor.item()
        if pred == 1:
            print("Analysis: Positive")
        else:
            print("Analysis: Negative")