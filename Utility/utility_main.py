# %%
import os

# %%
os.system("pipreqs . --force")

# %%

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)

os.system(f"pipreqs {dir_path}/../SentimentAnalysis/ --force")