import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def video_dict_get(name):
    f6 = open(r".\Data_audio_video\valid_sentiment.p",'rb')
    f5 = open(r".\Data_audio_video\valid_video.p",'rb')
    f7 = open(r".\Data_audio_video\\"+name+ "_video.p",'rb')
    data6 = pickle.load(f6)
    data5 = pickle.load(f5)
    data7 = pickle.load(f7)
    
    # 提取特征数据并创建 DataFrame
    features_data = []
    y_values = []
    for sample_name, features_array in data5.items():
        # 提取第一条所在行不为0的特征
        selected_features = features_array[np.any(features_array != 0, axis=1)]
        mean_features = np.mean(selected_features,axis = 0)
        mean_features = np.array(mean_features)
        new_mean_features = np.append(mean_features, data6[sample_name][0])
        features_data.append(new_mean_features)
        y_values.append(sample_name)
    
    
    
    df = pd.DataFrame(features_data, columns=[f'Feature_{i+1}' for i in range(features_data[0].shape[0])], index=y_values)
    df_corr = df.corr()
    
    threshold_high = 0.0618
    threshold_low = -0.05
    relevant_features = df_corr[(df_corr['Feature_714'] > threshold_high) | (df_corr['Feature_714'] < threshold_low)]
    relevant_feature_names = relevant_features.index.tolist()
    
    # 打印选取的相关特征的列名
    print(len(relevant_feature_names))
    origin_nochange_features = []
    for i in range(len(relevant_feature_names)):
        origin_nochange_features.append(int(relevant_feature_names[i][8:])-1)
        
    
    origin_nochange_features.pop()
    
    
    features_data = []
    
    features_data_num = []
    keys = []
    for sample_name, features_array in data7.items():
        # 提取第一条所在行不为0的特征
        selected_features = features_array[np.any(features_array != 0, axis=1)]
        features_data_num.append(selected_features.shape[0])
        keys.append(sample_name)
        for i in range(selected_features.shape[0]):
            features_data.append(selected_features[i][12:68])
    features_data = np.array(features_data)


    pca = PCA(n_components=14)

    
    # 创建 MinMaxScaler 对象
    scaler = MinMaxScaler()
    
    # 对数据进行归一化
    features_data0 = scaler.fit_transform(features_data)
    
    # 将数据进行 PCA 降维
    reduced_data = pca.fit_transform(features_data0)
    
    # 打印降维后的数据形状
    print(reduced_data.shape)
    
    
    features_data1 = []
    
    for sample_name, features_array in data7.items():
        # 提取第一条所在行不为0的特征
        selected_features = features_array[np.any(features_array != 0, axis=1)]
        for i in range(selected_features.shape[0]):
            features_data1.append(selected_features[i][298:366])
    features_data1 = np.array(features_data1)
    
    
    # 对数据进行归一化
    features_data1 = scaler.fit_transform(features_data1)
    pca = PCA(n_components=16)
    # 将数据进行 PCA 降维
    features_data10 = pca.fit_transform(features_data1)
    
    # 将数据进行 PCA 降维
    reduced_data1 = pca.fit_transform(features_data10)
    
    # 打印降维后的数据形状
    print(reduced_data1.shape)
    
    
    
    features_data2 = []
    
    for sample_name, features_array in data7.items():
        # 提取第一条所在行不为0的特征
        selected_features = features_array[np.any(features_array != 0, axis=1)]
        for i in range(selected_features.shape[0]):
            features_data2.append(selected_features[i])
    features_data2 = np.array(features_data2)
    
    
    # from sklearn.preprocessing import StandardScaler
    
    # 创建 StandardScaler 对象
    scaler = MinMaxScaler()
    
    # 对数据进行标准化
    
    
    features_data20 = features_data2[:, origin_nochange_features]
    reduced_data2 = scaler.fit_transform(features_data20)
    # 将提取的列合并为一个新的数组
    final_data = np.concatenate([reduced_data2,reduced_data1,reduced_data], axis=1)
    
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
    
    to_pickle(video_dicts, './'+name+'_video_dicts.pkl')
    
    return video_dicts
video_dict_get('test')
video_dict_get('valid')
video_dict_get('train')

all_s =['test','valid','train']
dicts = {}
for x in all_s:
    f = open( './'+x+'_video_dicts.pkl','rb')
    data = pickle.load(f)
    dicts.update(data)
print(len(dicts.keys()))

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

to_pickle(dicts,'video_dicts.pkl')