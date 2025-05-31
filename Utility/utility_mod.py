# %%
import multiprocessing, torch, struct, yaml

# %%
class Util:
    
    def read_yaml(yaml_path):
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def write_yaml(yaml_path, yaml_dict):
        with open(yaml_path, "w") as file:
            yaml.safe_dump(yaml_dict, file)

    def get_params(yaml_path):
        params = Util.read_yaml(yaml_path)
        params["cpu_threads"] = multiprocessing.cpu_count()
        params["arch_bit"] = struct.calcsize("P") * 8
        params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params["platform"] = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{params["platform"]=}")
        return params

    def increment_model_index(yaml_path):
        params = Util.read_yaml(yaml_path)
        params["model"]["index"] = params["model"]["index"]+1
        params = Util.write_yaml(yaml_path, params)
        return params