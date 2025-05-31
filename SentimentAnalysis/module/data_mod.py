# %%
import datasets, pprint, os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class SentimentDataModule:
# %%
    def __init__(self, params):
        self.params = params

# %%
    def fetch_dataset(self):
        print("fetch_dataset()")
        if self.params["platform"] == "pc":
            if os.path.isdir(self.params["raw_data_path"]):
                self.dataset = datasets.load_from_disk(self.params["raw_data_path"])
            else:
                self.dataset = datasets.load_dataset(self.params["dataset_name"])
                self.dataset.save_to_disk(self.params["raw_data_path"])
        elif self.params["platform"] == "collab":
            self.dataset = datasets.load_dataset(self.params["raw_data_path"])

        self.dataset = self.dataset.remove_columns("title")
        # Skipping data prep as it is not part of the judging criteria
        print(f"{len(self.dataset['train'])=}, {len(self.dataset['test'])=}")
        pprint.pprint(f"{self.dataset['train'][0]=}")
        return self

# %%
    def set_tokenizer(self):
        print("set_tokenizer()")
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["tokenizer"]["name"])

    def encode_dataset(self):
        print("encode_dataset()")
        tokenizer_params = self.params["tokenizer"]
        def tokenize_row(row):
            return self.tokenizer(row["content"], padding=tokenizer_params["padding"],
                                  max_length = tokenizer_params["max_length"],
                                  truncation=tokenizer_params["truncation"])

        self.encoded_dataset = self.dataset.map(tokenize_row, batched = True)
        self.encoded_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"]
        )
        return self
    
# %%
    def load_encoded_dataset(self):
        print("load_encoded_dataset()")
        if self.params["platform"] == "pc":
            if os.path.isdir(self.params["encoded_data_path"]):
                self.encoded_dataset = datasets.load_from_disk(self.params["encoded_data_path"])
            else:
                self.encode_dataset()
                self.encoded_dataset.save_to_disk(self.params["encoded_data_path"])
        else:
            self.encode_dataset()

# %%
    def get_data_loader(self):
        print("get_data_loader()")
        self.train_dataloader = DataLoader(self.encoded_dataset["train"], batch_size=self.params["arch_bit"], shuffle=True, num_workers=self.params["cpu_threads"]-2)
        self.val_dataloader = DataLoader(self.encoded_dataset["test"], batch_size=self.params["arch_bit"] , shuffle=False, num_workers=self.params["cpu_threads"]-2)
        return self.train_dataloader, self.val_dataloader
    

