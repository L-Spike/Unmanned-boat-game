import pickle
import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

description = 'draw graph'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('path', type=str, help='the path of pickle file')
args = parser.parse_args()
file_path = args.path

plt.figure(1)
# plt.subplot(1, 2, 1) #图一包含1行2列子图，当前画在第一行第一列图上
with open(os.path.join("train_data", file_path), "rb") as f:
    data = pickle.load(f)

# x1 = data["Cumulative reward"]

time = range(10)
sns.set(style="darkgrid")
plt.subplot(1,2,1)

data_diy = [[1,2,2],[4,5,6], [7,7,7],[9,8,9]]

data1 = data['Cumulative reward'][500:]
sns.lineplot( data=data_diy, color="r", estimator='mean')
# sns.tsplot(time=time, data=x2, color="b", condition="dagger")
plt.ylabel("cumulative reward")
plt.xlabel("episode Number")
plt.title("Results")

plt.subplot(1,2,2)
rewards1 = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
rewards2 = np.array([0, 0,0.1,0.4,0.5,0.5,0.55,0.8,0.9,1])
rewards=np.concatenate((rewards1,rewards2)) # 合并数组
rewards = np.vstack((rewards1,rewards2))

episode1=range(len(rewards1))
episode2=range(len(rewards2))
episode=np.concatenate((episode1,episode2))
episode = np.vstack((episode1,episode2))
df = pd.DataFrame(rewards).melt(var_name='episode',value_name='reward') # 推荐这种转换方法
print(df)
# sns.lineplot(x=episode,y=rewards)
print(episode,rewards)
plt.xlabel("episode")
plt.ylabel("reward")


# plt.subplot(1, 2, 1) #图一包含1行2列子图，当前画在第一行第一列图上
plt.figure(2)
df = sns.load_dataset("iris")
# print(df)
plt.subplot(1, 2, 1) #图一包含1行2列子图，当前画在第一行第一列图上
sns.lineplot( x = 'sepal_width', y = 'sepal_length', data=df, color="r")
plt.subplot(1, 2, 2) #图一包含1行2列子图，当前画在第一行第一列图上
sns.lineplot(data=df,)

# plt.figure(1)

# # sns.set(style="darkgrid", font_scale=1.5)
# data2 = data['losses'][500:]
# sns.lineplot( data=data2, color="r")
# # sns.tsplot(time=time, data=x2, color="b", condition="dagger")
# plt.ylabel("losses")
# plt.xlabel("episode Number")
# plt.title("Results")

plt.show()
