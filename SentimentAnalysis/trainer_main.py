# %%
import torch
import torch.nn as nn
import pandas as pd

# %%
import sys, os

# %%
# %load_ext autoreload
# %autoreload 2
utility_path = "../Utility"
sys.path.append(utility_path)

from Avrioc.SentimentAnalysis.module.data_mod import SentimentDataModule
from Avrioc.SentimentAnalysis.module.model_mod import LstmClassifier, TrainModel
from utility_mod import Util

# %%
yaml_path = "params.yaml"
util_yaml_path = "../Utility/params.yaml"

util_params = Util.get_params(util_yaml_path)
data_params = Util.get_params(yaml_path)["data"]
data_params.update(util_params)
model_params = Util.get_params(yaml_path)["model"]
model_params.update(util_params)

sdm = SentimentDataModule(data_params)

# %%
is_platform_pc = sdm.params["platform"] == "pc"
is_encoded_data_present = os.path.isdir(sdm.params["encoded_data_path"])

if not is_platform_pc or not is_encoded_data_present:
    sdm.fetch_dataset()

sdm.set_tokenizer()
sdm.load_encoded_dataset()

# %%
train_dataloader, val_dataloader = sdm.get_data_loader()
model = LstmClassifier(vocab_size = sdm.tokenizer.vocab_size, dim_list = model_params["dim_list"], dropout=model_params["dropout"])
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_params["optimizer_lr"])

train_model = TrainModel(model_params, train_dataloader, val_dataloader, model, loss_func, optimizer)
output = train_model.train_model()

# %%

# Util.write_yaml(output_path+"/config.yaml", model_params) 
print(model_params)


# %%
metrics_df = pd.DataFrame(output)
plt = train_model.plot_metrics(metrics_df)
Util.increment_model_index(yaml_path)
metrics_df = pd.DataFrame(output)
model_params["index"]+=1
output_path = f"output/model_{model_params['index']}"

os.mkdir(output_path)
torch.save(model.state_dict(), f"{output_path}/model.pt")
plt.savefig(f"{output_path}/loss_accuracy_plot.png")
del model_params['device']
Util.write_yaml(output_path+"/config.yaml", model_params) 


# %%
# To do
# Generalization
# Config the model
# Parametrization
# Split files for data clean mod
# Visualize the model
# Unit test?
# Implement Early Stop
# Data EDA
# Remove comments


