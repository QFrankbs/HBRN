import sys
import os
import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict
from subprocess import check_call
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# sys.path.append("../..") 
# from new_config import parse_args
# from utils.compute_args import compute_args

lang_seq_len = 60
import torch
import h5py
mosei_unalign_hdf5 = r"F:\MOSEI_data\mosei_unalign.hdf5"
def print_string_data(dataset, name,name1, level=0):
    global results
    data = dataset[()].astype('U32')  # 解码为字符串
    new_data = ''
    for i in range(len(data)):
        if data[i][0]!='sp':
            new_data = new_data+' '+data[i][0]
    # print("  " * level + f"Dataset: {name}")
    # print("  " * level + f"Data:\n{new_data}")
    # print("\n")
    results[name1]=new_data

def identify_hdf5_structure(group, level=0):
    global results
    if level==3:
        return 0
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            # print("  " * level + f"Group: {name}")
            identify_hdf5_structure(item, level + 1)
        elif isinstance(item, h5py.Dataset):
            data_type = item.dtype
            # shape = item.shape
            # print("  " * level + f"Dataset: {name} ({data_type}, shape={shape})")
            if data_type == '|S32':
                print_string_data(item, name, level)
ids = []
sentiments=[]
emotions = []

def identify_hdf5_structure_new(group, level=0):
    global ids
    global results
    global sentiments
    global emotions
    
    if level==3:
        return 0
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            # print("  " * level + f"Group: {name}")
            idx = [name]
            for name1, new_item in item.items():
                if isinstance(new_item, h5py.Dataset):
                    # print(new_item)
                    data_type = new_item.dtype
                    if data_type == 'f8':
                        data = new_item[()].astype('f8')
                        # print(type(data))
                        idx = idx + list(data[0])
                        ids.append(np.array(idx))
                        # print(data)
                    if data_type == 'f4':
                        data = new_item[()].astype('f4')
                        sentiments.append(np.array([data[0][0]]))
                        emotions.append(np.array(data[0][1:]))
                    # print("  " * level + f"Dataset: {name1} ({data_type}, shape={shape})")
                    if data_type == '|S32':
                        print_string_data(new_item, name1, name, level)



# 打开HDF5文件
file = h5py.File(mosei_unalign_hdf5, 'r')
resut=[]
# 调用函数开始识别
identify_hdf5_structure_new(file['All Labels'])
# print(sentiments[0])
# print(emotions[0])
sentiments_dict = {ids[i][0]: sentiments[i] for i in range(len(ids))}
emotions_dict = {ids[i][0]: emotions[i] for i in range(len(ids))}


# args = compute_args(parse_args())

from transformers import BertTokenizer, BertModel
def get_length(x):
    return x.shape[1] - (np.sum(x, axis=-1) == 0).sum(1)

pickle_filename = './mosei_data.pkl'

with open(pickle_filename, 'rb') as f:
    d = pickle.load(f)




train_split_noalign = d['train']
dev_split_noalign = d['valid']
test_split_noalign = d['test']


mosei_hdf5 = r"F:\MOSEI_data\mosei.hdf5"
affect_data = h5py.File(mosei_hdf5, 'r')
print(affect_data.keys())

def get_rawtext(path, data_kind, vids):
    """Get raw text modality.

    Args:
        path (str): Path to h5 file
        data_kind (str): String for data format. Should be 'hdf5'.
        vids (list): List of video ids.

    Returns:
        tuple(list,list): Tuple of text_data and video_data in lists.
    """
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
        text_data = []
        new_vids = []
        for vid in vids:
            text = []
            # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
            # vid_id = '{}[{}]'.format(id, seg)
            vid_id = vid
            try:
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            except:
                print("missing", vid, vid_id)
        return text_data, new_vids
    else:
        print('Wrong data kind!')

WORD = 'words'
keys = list(affect_data[WORD].keys())
print(len(keys))

raw_text, vids = get_rawtext(
    'mosei.hdf5', 'hdf5', keys)

another_dict = {vids[i]: raw_text[i] for i in range(len(vids))}
# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 1e-6

def pad_feature(feat, max_len):
    if feat.shape[0] > max_len:
        feat = feat[feat.shape[0]-max_len:]

    feat = np.pad(
        feat,
        ((0, max_len - feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return feat

# place holders for the final train/dev/test dataset
train = []
dev = []
test = []
f = open(r"video_dicts.pkl",'rb')
video_dicts = pickle.load(f)
f = open(r"audio_dicts.pkl",'rb')
audio_dicts = pickle.load(f)
# define a regular expression to extract the video ID out of the keys
num_drop = 0  # a counter to count how many data points went into some processing issues
model_name = 'bert-base-uncased'  # 选择合适的预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

if True:
    t = np.concatenate(
        (train_split_noalign['text'], dev_split_noalign['text'], test_split_noalign['text']), axis=0)
    tlens = get_length(t)
    
    v = np.concatenate(
        (train_split_noalign['vision'], dev_split_noalign['vision'], test_split_noalign['vision']), axis=0)
    vlens = get_length(v)

    a = np.concatenate(
        (train_split_noalign['audio'], dev_split_noalign['audio'], test_split_noalign['audio']), axis=0)
    alens = get_length(a)

    emotion_label = np.concatenate(
        (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)
    
    sentiment_label = np.concatenate(
        (train_split_noalign['labels'], dev_split_noalign['labels'], test_split_noalign['labels']), axis=0)
    
    L_T = t.shape[1]
    L_V = v.shape[1]
    L_A = a.shape[1]

all_id = np.concatenate((train_split_noalign['id'], dev_split_noalign['id'], test_split_noalign['id']),
                        axis=0)

def check_and_modify_data(data_list, new_data):
    for j, existing_data in enumerate(data_list):
        if new_data[0] in existing_data[0]:
            if list(existing_data[1:]) == list(new_data[1:]):
                return data_list[j][0]
    return False
sentences=[]
new_all_id = []
new_emotion_label = []
new_sentiment_label = []
for i in range(len(all_id)):
    judge = check_and_modify_data(ids, all_id[i])
    # print(judge)
    print(i)
    if judge!=False:
        new_all_id.append(judge)
        new_emotion_label.append(emotions_dict[judge])
        new_sentiment_label.append(sentiments_dict[judge])
        sentences.append(another_dict[judge])
    else:
        print(judge)
all_id = np.array(new_all_id)
emotion_label = np.array(new_emotion_label)
sentiment_label = np.array(new_sentiment_label)
all_id_list = all_id.tolist()
train_size = len(train_split_noalign['id'])
dev_size = len(dev_split_noalign['id'])


dev_start = train_size
test_start = train_size + dev_size
def pad_feature(feat, max_len):
    if feat.shape[0] > max_len:
        feat = feat[:max_len]

    feat = np.pad(
        feat,
        ((max_len-feat.shape[0], 0), (0, 0)),
        mode='constant',
        constant_values=0
        )
    return feat


##做vid 和 cid之间的映射关系
for i, idd in enumerate(all_id_list):
    # get the video ID and the features out of the aligned dataset

    # matching process
    try:
        index = i
    except:
        import ipdb
        ipdb.set_trace()
    """
        Retrive noalign data from pickle file
    """
    _words = sentences[index]
    
    _text = t[i]
    _id = all_id[i]
    _visual = video_dicts[_id]
    _acoustic = audio_dicts[_id]
    _tlen = tlens[i]
    _vlen = vlens[i]
    _alen = alens[i]

    _sentiment = sentiment_label[i]
    _emotion = emotion_label[i]

    # remove nan values
    _text = np.nan_to_num(_text)
    _visual = np.nan_to_num(_visual)
    _acoustic = np.nan_to_num(_acoustic)
    texts = []
    visual = []
    acoustic = []

    """TODO: Add length counting for other datasets
    """
    texts = _text[L_T - _tlen:, :]
    input_ids = tokenizer.encode(_words, add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # 转换为PyTorch张量并增加批次维度
    # 获取BERT模型的输出
    with torch.no_grad():
        outputs = model(input_ids)
    
    # 提取句子的BERT词向量
    sentence_embedding = outputs[0].squeeze(0)
    
    # labels.append(torch.from_numpy(sample[1]))
    
    required_cols = sentence_embedding.shape[1]

    # 创建一个零矩阵，形状为 (50, required_cols)
    language1 = np.zeros((_text.shape[0], required_cols))
    
    # print(language1.shape)
    # 将 _text 复制到零矩阵的左侧，以填充
    language1[:, :_text.shape[1]] = _text
    # print(language1.shape)
    
    language2 = np.concatenate((language1, sentence_embedding), axis=0)
    language3 = pad_feature(language2, lang_seq_len)
    print(i)
    visual = _visual
    # visual = pad_sequence(visual, target_len=args.video_seq_len)
    acoustic = _acoustic
    # acoustic = pad_sequence(acoustic, target_len=args.audio_seq_len)
    # z-normalization per instance and remove nan/infs
    # visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
    # acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))
    ### 这一部分数据分为训练数据和验证数据
    if i < dev_start:
        train.append((i,language3, visual, acoustic, _sentiment,_emotion))

        # train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label0, _source0, idd))
        # train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label1, _source1, idd))
    elif i >= dev_start and i < test_start:
        dev.append((i,language3, visual, acoustic,_sentiment,_emotion))
        # dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label0, 0, idd))
    elif i >= test_start:
        test.append((i,language3, visual, acoustic,_sentiment,_emotion))
        # test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label0, 0, idd))
    else:
        print(f"Found video that doesn't belong to any splits: {idd}")
    # if i < dev_start:
    #     train.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
    # elif i >= dev_start and i < test_start:
    #     dev.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
    # elif i >= test_start:
    #     test.append(((words, visual, acoustic, actual_words, _vlen, _alen), _label, idd))
    # else:
    #     print(f"Found video that doesn't belong to any splits: {idd}")

print(f"Total number of {num_drop} datapoints have been dropped.")
print("Dataset split")
print("Train Set: {}".format(len(train)))
print("Validation Set: {}".format(len(dev)))
print("Test Set: {}".format(len(test)))

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
# Save glove embeddings cache too
# self.pretrained_emb = pretrained_emb = load_emb(word2id, config.word_emb_path)
# torch.save((pretrained_emb, word2id), CACHE_PATH)

# Save pickles
# to_pickle(train, './new_train_align_v4_0424.pkl')
# to_pickle(dev, './new_dev_align_v4_0424.pkl')
# to_pickle(test, './new_test_align_v4_0424.pkl')

to_pickle(train, './train_align.pkl')
to_pickle(dev, './valid_align.pkl')
to_pickle(test, './test_align.pkl')