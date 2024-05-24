import argparse
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from skimage import io, transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import sys
import time

class Rellis3DAnnotaionDataset(Dataset):

  def __init__(self, root_path, sample_size=None, train=False) -> None:
    super().__init__()

    self.root_path = root_path

    self.target_path = join(self.root_path, "annotation")

    if sample_size:
      self.inputs = pd.read_csv(join(self.root_path, "train.txt" if train else "val.txt"), header=None)[0].values.tolist()[:sample_size]
    else:
      self.inputs = pd.read_csv(join(self.root_path, "train.txt" if train else "val.txt"), header=None)[0].values.tolist()
    # self.targets = [f for f in listdir(self.target_path) if isfile(join(self.target_path, f))]

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    img_path_tgt_patch = join(self.target_path, self.inputs[idx] + ".png")
    mask = io.imread(img_path_tgt_patch)

    IDs =    [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34]
    Groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    return mask

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Dataset Analyzer")
  parser.add_argument("path", help="The path to the annotation data.")
  parser.add_argument("sample_size", help="The size of the sample to test", type=int)
  args = parser.parse_args()

  dataset = Rellis3DAnnotaionDataset(args.path, train=True, sample_size=args.sample_size)

  df = pd.Series()

  for idx, (mask) in enumerate(dataset):
    print("Loaded mask ", idx)
    
    mask_df = pd.DataFrame(mask.reshape(-1))[0]
    mask_df = mask_df.value_counts()
    df = df.add(mask_df, fill_value=0)

  print("------------------------------------------")
  print(df)
  print("------------------------------------------")
  print("")

  df = df.rename(index={0: "void", 1: "dirt", 3: "grass", 4:"tree", 5:"pole", 6:"water", 7:"sky", 8:"vehicle", 
            9:"object", 10:"asphalt", 12:"building", 15:"log", 17:"person", 18:"fence", 19:"bush", 
            23: "concrete", 27:"barrier", 31:"puddle", 33:"mud", 34:"rubble"})
  df = df.sort_values(ascending=False)
  inset_df = df[8:]
  main_df = df[:8]
  
  ax = main_df.plot.bar(x='lab', y='val', rot=0)
  fig = ax.figure
  
  left, bottom, width, height = [0.65, 0.6, 0.2, 0.2]
  ax2 = fig.add_axes([left, bottom, width, height])
  
  inset_df.plot(ax=ax2, kind="bar")
  ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
  fig.savefig("plot.png", bbox_inches="tight")

  print("saved plot.png")
  
