# %%
import os, sys, torch

# %%
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = os.getcwd() 

module_path = os.path.join(current_dir, "module")
sys.path.append(module_path)

from data_mod import SentimentDataModule
from model_mod import LstmClassifier, TrainModel


utility_path = os.path.join(current_dir, "../Utility")
sys.path.append(utility_path)

from utility_mod import Util

# %%
yaml_path = os.path.join(current_dir, "params.yaml")
util_yaml_path = os.path.join(current_dir, "../Utility/params.yaml")

# util_params = Util.get_params(util_yaml_path)
data_params = Util.get_params(yaml_path)["data"]
# data_params.update(util_params)
model_params = Util.get_params(yaml_path)["model"]
# model_params.update(util_params)

sdm = SentimentDataModule(data_params)

# %%
sdm.set_tokenizer()
model = LstmClassifier(vocab_size = sdm.tokenizer.vocab_size, dim_list = model_params["dim_list"], dropout=model_params["dropout"])
model_state = torch.load(model_params["inference_model_path"], map_location="cpu")
model.load_state_dict(model_state)
model.eval()

# %%
def predict(text):
    tokenizer_params = sdm.params["tokenizer"]
    # tokenizer_params["return_tensors"]
    token_id = sdm.tokenizer.encode(text, padding=tokenizer_params["padding"],
                                  max_length = tokenizer_params["max_length"],
                                  truncation=tokenizer_params["truncation"])
    input_tensor = torch.tensor(token_id).unsqueeze(0)
    print(f"{input_tensor=}, {type(input_tensor)}")
    with torch.no_grad():
        logit_tensor = model(input_tensor).view(-1)
    
    print(f"{logit_tensor=}, {type(logit_tensor)}")
    pred_tensor = (torch.sigmoid(logit_tensor) > 0.5).long()
    print(f"{pred_tensor=}, {type(pred_tensor)}")
    # probs = torch.softmax(pred.logits, dim=1)
    # predicted_class = torch.argmax(probs, dim=1).item()
    # return predicted_class, probs.numpy()

predict("It's a great day")

# %%
# is_platform_pc = sdm.params["platform"] == "cuda"
# is_encoded_data_present = os.path.isdir(sdm.params["encoded_data_path"])

# if not is_platform_pc or not is_encoded_data_present:
#     sdm.fetch_dataset()

# sdm.set_tokenizer()
# sdm.load_encoded_dataset()


