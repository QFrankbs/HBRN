import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def video_dict_get(name):
    f7 = open(r".\Data_audio_video\\"+name+ "_audio.p",'rb')
    data7 = pickle.load(f7)
    
    features_data = []
    
    features_data_num = []
    keys = []
    for sample_name, features_array in data7.items():
        # 提取第一条所在行不为0的特征
        selected_features = features_array[np.any(features_array != 0, axis=1)]
        features_data_num.append(selected_features.shape[0])
        keys.append(sample_name)
        for i in range(selected_features.shape[0]):
            features_data.append(selected_features[i][0:1])
    features_data = np.array(features_data)
    
    scaler = MinMaxScaler()
    
    reduced_data = scaler.fit_transform(features_data)
    print(reduced_data.shape)
    
    
    features_data1 = []
    
    for sample_name, features_array in data7.items():
        # 提取第一条所在行不为0的特征
        selected_features = features_array[np.any(features_array != 0, axis=1)]
        for i in range(selected_features.shape[0]):
            features_data1.append(selected_features[i][1:74])
    reduced_data1 = np.array(features_data1)


    where_are_nan = np.isnan(reduced_data1)
    where_are_inf = np.isinf(reduced_data1)
    #nan替换成0,inf替换成nan
    reduced_data1[where_are_nan] = 0
    reduced_data1[where_are_inf] = 0

    
    pca = PCA(n_components=63)
    
    reduced_data1 = pca.fit_transform(reduced_data1)
    print(reduced_data1.shape)
    
    final_data = np.concatenate([reduced_data,reduced_data1], axis=1)
    
    video_dicts={}
    def pad_feature(feat, max_len):
        if feat.shape[0] > max_len:
            feat = feat[max_len-feat.shape[0]:]
    
        feat = np.pad(
            feat,
            ((max_len-feat.shape[0], 0), (0, 0)),
            mode='constant',
            constant_values=0
        )
    
        return feat
    
    
    start = 0
    for i in range(len(features_data_num)):
        video_dicts[keys[i]]=pad_feature(final_data[start:start+features_data_num[i]],60)
        start+=features_data_num[i]
    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    
    to_pickle(video_dicts, './'+name+'_audio_dicts.pkl')
    
    return video_dicts
video_dict_get('test')
video_dict_get('valid')
video_dict_get('train')

all_s =['test','valid','train']
dicts = {}
for x in all_s:
    f = open( './'+x+'_audio_dicts.pkl','rb')
    data = pickle.load(f)
    dicts.update(data)
print(len(dicts.keys()))

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

to_pickle(dicts,'audio_dicts.pkl')