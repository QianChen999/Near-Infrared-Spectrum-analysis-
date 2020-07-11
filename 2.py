# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:44:56 2020

@author: Qian Chen
"""


import numpy as np
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
import matplotlib.pyplot as plt
import pandas as pd



import os


#%%
os.chdir(r"D:\iPython-space\数据挖掘题目\近红外试题2\建模集光谱")
loadfile = os.listdir()

data = pd.read_csv(loadfile[0], header = None, names = ['波长', '吸光度'])
data['样本序号'] = loadfile[0].split('.')[0]

for i in loadfile[1:]:
    df = pd.read_csv(i, header=None, names = ['波长', '吸光度'])
    df['样本序号'] = i.split('.')[0]
    data = pd.concat([data, df])

#%%
    
data_2=pd.pivot_table(data,index='样本序号',columns='波长',values='吸光度')
data_2.reset_index(inplace=True)
data_2['样本序号'] = data_2['样本序号'].astype(int)
#%%

data_3 = data_2.groupby('样本序号').mean().copy()
data_3.reset_index(inplace=True)

#%%



os.chdir(r"D:\iPython-space\数据挖掘题目\近红外试题2")
y = pd.read_excel('建模集对应值.xlsx')
#%%

model_data = pd.merge(y, data_3, on= '样本序号')
#%%

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

target = model_data['建模集对应值']
data_4 = model_data.iloc[:, 2:]

train_data, test_data, train_target, test_target = train_test_split(
    data_4, target, test_size=0.4, train_size=0.6, random_state=123) 

print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (train_data.shape[0], test_data.shape[0]))

#%%
from keras import models
from keras import layers

def build_model():
    '''
    Because you’ll need to instantiate
    the same model multiple times, you
    use a function to construct it.
    '''
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


#%%

# K-fold validation


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    '''
    Prepares the validation data: data from partition #k
    '''
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_target[i * num_val_samples: (i + 1) * num_val_samples]

    '''
    Prepares the training data: data from all other partitions
    '''
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_target[:i * num_val_samples],
         train_target[(i + 1) * num_val_samples:]], axis=0)

    '''
    Builds the Keras model (already compiled)
    '''
    model = build_model()
    '''
    Trains the model (in silent mode, verbose = 0)
    '''
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
              batch_size=1, verbose=0)
    '''
    Evaluates the model on the validation data
    '''
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)


#%%
all_scores
np.mean(all_scores)
#%%

# Saving the validation logs at each fold
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    '''
    Prepares the validation data: data from partition #k
    '''
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_target[i * num_val_samples: (i + 1) * num_val_samples]

    '''
    Prepares the training data: data from all other partitions
    '''
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_target[:i * num_val_samples],
         train_target[(i + 1) * num_val_samples:]], axis=0)

    '''
    Builds the Keras model (already compiled)
    '''
    model = build_model()
    '''
    Trains the model (in silent mode, verbose = 0)
    '''
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epochs,
                        validation_data=(val_data, val_targets),
                        batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
#%%

#Building the history of successive mean K-fold validation scores
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]    

#%%

#Plotting validation scores


plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#%%

#Plotting validation scores, excluding the first 10 data points

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#%%
#Training the final model

#Gets a fresh, compiled model
model = build_model()

#Trains it on the entirety of the data
model.fit(train_data, train_target, epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_target)

print ("test_mse_score：%d, test_mae_score：%d" % (test_mse_score, test_mae_score))
#%%


#%%
# 预测验证集

os.chdir(r"D:\iPython-space\数据挖掘题目\近红外试题2\验证集光谱")
loadfile = os.listdir()

data = pd.read_csv(loadfile[0], header = None, names = ['波长', '吸光度'])
data['样本序号'] = loadfile[0].split('.')[0]

for i in loadfile[1:]:
    df = pd.read_csv(i, header=None, names = ['波长', '吸光度'])
    df['样本序号'] = i.split('.')[0]
    data = pd.concat([data, df])

    
    
    
#%%
x_predict =pd.pivot_table(data,index='样本序号',columns='波长',values='吸光度')
x_predict = x_predict.drop_duplicates()


#model_test = pd.merge(Y, test, on= '样本序号')

#%%
#用模型预测

y_pred = model.predict(x_predict,batch_size = 1)
