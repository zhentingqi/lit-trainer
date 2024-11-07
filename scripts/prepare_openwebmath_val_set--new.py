import os


src_dir = "/n/holyscratch01/glassman_lab/Users/zhentingqi/big_data/HF_datasets/openwebmath/whole_data"
tgt_dir = "/n/holyscratch01/glassman_lab/Users/zhentingqi/big_data/HF_datasets/openwebmath/eval_500M"
os.makedirs(tgt_dir, exist_ok=True)

# Count the number of files in the source directory
files = os.listdir(src_dir)
num_files = len(files)

# Select 1/30 of the files to copy to the target directory
num_files_to_copy = num_files // 30

from random import sample
selected_files = sample(files, num_files_to_copy)

from shutil import copyfile
from tqdm import tqdm
for file in tqdm(selected_files):
    src_file = os.path.join(src_dir, file)
    tgt_file = os.path.join(tgt_dir, file)
    copyfile(src_file, tgt_file)
