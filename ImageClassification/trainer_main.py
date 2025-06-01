# %%
import os, sys
import torch
import torch.nn as nn
import torchvision as torchv
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# %%
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except:
    current_dir = os.getcwd() 

module_path = os.path.join(current_dir, "module")
sys.path.append(module_path)

from data_mod import ImageDataModule
from model_mod import Resnet18Classifier, TrainModel

utility_path = os.path.join(current_dir, "../Utility")
sys.path.append(utility_path)

from utility_mod import Util

# %%
yaml_path = os.path.join(current_dir, "params.yaml")
data_params = Util.get_params(yaml_path)
data_params.update(data_params["data"])
model_params = Util.get_params(yaml_path)
model_params.update(data_params["model"])

idm = ImageDataModule(data_params)

# %%
idm.data_path = os.path.join(current_dir, "data")

idm.set_input_transform()
idm.fetch_dataset()
train_dataloader, val_dataloader = idm.get_data_loader(percent_data = data_params["data_percent"])

# %%
len(val_dataloader)

# %%
# for batch in train_dataloader:
#     img_tensor = batch[0].to(model_params["device"])
#     label_tensor = batch[1].to(model_params["device"])
#     print(f"{img_tensor.shape=} {label_tensor.shape=}")
# model_params.keys()

# %%
# train_dataloader, val_dataloader = sdm.get_data_loader(percent_data = data_params["data_percent"])
model = Resnet18Classifier(model_params["dim_list"], model_params["dropout"], model_params["num_groups"])
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_params["optimizer_lr"])

train_model = TrainModel(model_params, train_dataloader, val_dataloader, model, loss_func, optimizer)
output = train_model.train_model()

# %%
import pandas as pd
metrics_df = pd.DataFrame(output)
plt = train_model.plot_metrics(metrics_df)
Util.increment_model_index(yaml_path)
metrics_df = pd.DataFrame(output)
model_params["index"]+=1

output_path = f"{current_dir}/output/resnet18_model_{model_params['index']}"

os.mkdir(output_path)
torch.save(model.state_dict(), f"{output_path}/model.pt")
plt.savefig(f"{output_path}/loss_accuracy_plot.png")
del model_params['device']
Util.write_yaml(output_path+"/config.yaml", model_params) 



