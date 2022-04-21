import pickle
import os
import sys
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

description = 'draw graph'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', type=str, help='the path of pickle file')
args = parser.parse_args()
file_path = args.path

with open(os.path.join("train_data", file_path), "rb") as f:
    data = pickle.load(f)

# x1 = data["Cumulative reward"]

time = range(10)

sns.set(style="darkgrid", font_scale=1.5)
sns.lineplot( data=data, color="r")
# sns.lineplot( data=data, color="r")
# sns.tsplot(time=time, data=x2, color="b", condition="dagger")

plt.ylabel("cumulative reward")
plt.xlabel("episode Number")
plt.title("Results")

plt.show()
