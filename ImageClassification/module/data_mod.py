# %%
import os
from torch.utils.data import DataLoader, Subset
import torchvision as torchv

class ImageDataModule:
# %%
    def __init__(self, params):
        self.params = params
# %%
    def set_input_transform(self):
        self.input_transform = torchv.transforms\
            .Compose([torchv.transforms.Resize(tuple(self.params["transform"]["resize"])),
                      torchv.transforms.Grayscale(self.params["transform"]["grayscale"]),
                      torchv.transforms.ToTensor(),
                      torchv.transforms.Normalize(self.params["transform"]["normalize"]["mu"],
                                                  self.params["transform"]["normalize"]["sigma"]),
                      ])#3x244x244


# %%
    def fetch_dataset(self):
        print("fetch_dataset()")
        download = not os.path.isdir(self.data_path)
        print(f"{download=}")
        
        self.train_dataset = torchv.datasets.MNIST(self.data_path+"/train", train = True, download = download, transform=self.input_transform)
        self.test_dataset = torchv.datasets.MNIST(self.data_path+"/test", train = False, download = download, transform=self.input_transform)
        # Skipping data prep as it is not part of the judging criteria
        print(f"{len(self.train_dataset)=}, {len(self.test_dataset)=}")
        print(f"{self.train_dataset[0][0].shape=}")
        return self

# %%
    def get_data_loader(self, percent_data):
        print("get_data_loader()")
        train_count = int(len(self.train_dataset)*percent_data/100)
        test_count = int(len(self.test_dataset)*percent_data/100)
        
        train_subset = Subset(self.train_dataset, indices=range(train_count))
        test_subset = Subset(self.test_dataset, indices=range(test_count))
        self.train_dataloader = DataLoader(train_subset,
                                           batch_size=self.params["arch_bit"], shuffle=True,
                                           num_workers=self.params["cpu_threads"]-2)
        self.val_dataloader = DataLoader(test_subset,
                                         batch_size=self.params["arch_bit"] , shuffle=False,
                                         num_workers=self.params["cpu_threads"]-2)
        return self.train_dataloader, self.val_dataloader
    

