# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from transformers import BertModel
import torchvision as torchv

# %%
class Resnet18Classifier(nn.Module):
    def __init__(self, dim_list, dropout, num_groups):
        super().__init__()
        def group_normal_layer(num_channels):
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        self.resnet = torchv.models.resnet18(pretrained=False, norm_layer=group_normal_layer)#[ChannelHeightLength]
        self.resnet.fc = nn.Sequential(nn.Dropout(dropout),
                                       nn.Linear(dim_list[0], dim_list[1]))#[BatchHidden]

    def forward(self, img_tensor):
        logit = self.resnet(img_tensor)
        return logit
# %%
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

    def forward(self, input_id_tensor, attention_mask_tensor = None):
        logit = self.embedding(input_id_tensor)#[BatchTokenEmbedding]
        output, (hn, cn) = self.lstm(logit)#hn=[LayerBatchHidden]
        h = hn[-1]#[BatchHidden]
        logit = self.dropout(h)
        logit = self.linear(logit)#[BatchHidden]
        return logit
    
# %%
# class TransformerClassifier(nn.Module):
#     def __init__(self, dropout):
#         super().__init__()

#         self.encoder = BertModel.from_pretrained("bert-base-uncased")
#         self.dropout = nn.Dropout(dropout)
#         self.linear = nn.Linear(self.encoder.config.hidden_size, 1)

#     def forward(self, input_id_tensor, attention_mask_tensor):
#         output = self.encoder(input_ids=input_id_tensor, attention_mask=attention_mask_tensor)
#         pooled_h = output.pooler_output
#         logit = self.dropout(pooled_h)
#         logit = self.linear(logit)
#         return logit



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
        img_tensor = batch[0].to(self.params["device"])
        label_tensor = batch[1].to(self.params["device"])

        logit_tensor = self.model(img_tensor)#[BatchHidden]
        
        
        loss = self.loss_func(logit_tensor, label_tensor)
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

        for batch_num, batch in enumerate(dataloader):
            print(f"\rbatch_num={batch_num}/{len(dataloader)-1}", end="")
            if mode == "train":
                self.optimizer.zero_grad()

            logit_tensor, label_tensor, batch_loss = self.model_iteration(batch)
            
            total_loss+=batch_loss.item()
            total_pred += label_tensor.size(0)
            prob_tensor = torch.softmax(logit_tensor, dim=1)
            pred_tensor = prob_tensor.argmax(dim=1)
            
            correct_pred += (pred_tensor == label_tensor).sum().item()
            
            if mode == "train":
                batch_loss.backward()
                self.optimizer.step()
            
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
            with torch.no_grad():
                val_loss, val_accuracy = self.model_epoch("val")
            print(f"epoch={epoch}/{self.params['epoch']}|{train_loss=}|{val_loss=}")
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
    def __init__(self, model):
        self.model = model

    # %%
    def predict(self, img_tensor):
        # tokenizer_params["return_tensors"]
        with torch.no_grad():
            logit_tensor = self.model(img_tensor)
        
        
        prob_tensor = torch.softmax(logit_tensor, dim=1)
        pred_tensor = prob_tensor.argmax(dim=1)
        print(f"Detected: {pred_tensor.item()}")