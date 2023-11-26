from __future__ import print_function
import os
import pickle
import numpy as np
import torch
# from utils.plot import plot
from utils.tokenize import tokenize, create_dict, sent_to_ix, cmumosei_2, cmumosei_7, pad_feature
from transformers import BertTokenizer, BertModel

from torch.utils.data import Dataset
model_name = 'bert-base-uncased'  # 选择合适的预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
#from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Mosei_Dataset(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(Mosei_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test', 'private']
        self.name = name
        self.args = args
        self.private_set = name == 'private'
        self.dataroot = os.path.join(dataroot,'MOSEI')
        
        file = os.path.join(self.dataroot, name + "_align.pkl")
        self.datasets = pickle.load(open(file, "rb"))
        self.idx = [x[0] for x in self.datasets]
        self.language = [x[1] for x in self.datasets]
        self.video = [x[2] for x in self.datasets]
        self.audio = [x[3] for x in self.datasets]
        self.sentiment = [x[4] for x in self.datasets]
        self.emotion = [x[5] for x in self.datasets]
        # self.idx_sentiment = [x[6]-self.datasets[0][0] for x in self.datasets]
        self.length = len(self.emotion)
        
        
        # Creating embeddings and word indexes
        # print(len(self.key_to_word))

        # self.key_to_sentence = tokenize(self.key_to_word)  #用于清洗无关词汇或标点符号
        # print(len(self.key_to_sentence))
        # if token_to_ix is not None:
            # self.token_to_ix = token_to_ix
        # else: # Train
            # self.token_to_ix, self.pretrained_emb = create_dict(self.key_to_sentence, self.dataroot)
            # print(type(self.token_to_ix),type(self.pretrained_emb))
            # print(self.pretrained_emb.shape)
            # print(self.token_to_ix)
        # self.vocab_size = len(self.token_to_ix)
        
        self.l_max_len = args.lang_seq_len
        self.a_max_len = args.audio_seq_len
        self.v_max_len = args.video_seq_len
        
    def __getitem__(self, idx):
        def pad_sequence(sequences, target_len=-1, batch_first=False, padding_value=0.0):
            if target_len < 0:
                max_size = sequences[0].size()
                trailing_dims = max_size[1:]
            else:
                max_size = target_len
                trailing_dims = sequences[0].size()[1:]

            max_len = max([s.size(0) for s in sequences])
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims

            out_tensor = sequences[0].new_full(out_dims, padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        def Y_to_new_Y(idx):
            if self.args.task == "sentiment" and self.args.task_binary:
                Y = self.sentiment[idx]
                c = cmumosei_2(Y)
                y = np.array(c)
            if self.args.task == "sentiment" and not self.args.task_binary:
                Y = self.sentiment[idx]
                c = cmumosei_7(Y)
                y = np.array(c)
            if self.args.task == "emotion":
                Y = self.emotion[idx]
                Y[Y > 0] = 1
                y = Y
            return y
        
        L = pad_feature(self.language[idx],self.args.lang_seq_len)
        V = pad_feature(self.video[idx],self.args.video_seq_len)
        A = pad_feature(self.audio[idx],self.args.audio_seq_len)

        idx_negative = idx
        if self.args.task =='sentiment':
            while idx_negative == idx or Y_to_new_Y(idx_negative)==Y_to_new_Y(idx):
                idx_negative = np.random.randint(0, self.length)
        else:
            while idx_negative == idx or Y_to_new_Y(idx_negative).all()==Y_to_new_Y(idx).all():
                idx_negative = np.random.randint(0, self.length)

        L1 = pad_feature(self.language[idx_negative],self.args.lang_seq_len)
        y = Y_to_new_Y(idx)
        return idx, torch.from_numpy(L), torch.from_numpy(A), torch.from_numpy(V), torch.from_numpy(L1), torch.from_numpy(y)

    def __len__(self):
        return self.length