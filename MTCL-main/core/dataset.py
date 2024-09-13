'''
* @name: dataset.py
* @description: Dataset loading functions. Note: The code source references MMSA (https://github.com/thuiar/MMSA/tree/master).
'''

import json
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        self.args_dict = vars(args)
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims  #[t[1368,39,768],a[1368,400,33],v[1368,55,709]]
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        if self.args_dict.get('use_bert', None):
            self.text = data[self.mode]['text_bert'].astype(np.float32)
            self.args.feature_dims[0] = 768
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
            self.args.feature_dims[0] = self.text.shape[2]
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.args.feature_dims[1] = self.audio.shape[2]
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.args.feature_dims[2] = self.vision.shape[2]
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args_dict.get('need_normalized'):
            self.__normalize()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()


    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        sample['audio_lengths'] = self.audio_lengths[index]
        sample['vision_lengths'] = self.vision_lengths[index]
        return sample


def MMDataLoader(args,shuffle=True):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=shuffle)
        for ds in datasets.keys()
    }
    
    return dataLoader
max_len=50
def pad_collate(batch):
    (x_t, x_a, x_v, y_t, y_a, y_v, y_m) = zip(*batch)
    x_t = torch.stack(x_t, dim=0)
    x_v = torch.stack(x_v, dim=0)
    y_t = torch.tensor(y_t)
    y_a = torch.tensor(y_a)
    y_v = torch.tensor(y_v)
    y_m = torch.tensor(y_m)
    x_a_pad = pad_sequence(x_a, batch_first=True, padding_value=0)
    len_trunc = min(x_a_pad.shape[1], max_len)
    x_a_pad = x_a_pad[:, 0:len_trunc, :]
    len_com = max_len - len_trunc
    zeros = torch.zeros([x_a_pad.shape[0], len_com, x_a_pad.shape[2]], device='cpu')
    x_a_pad = torch.cat([x_a_pad, zeros], dim=1)

    return x_t, x_a_pad, x_v, y_t, y_a, y_v, y_m

def MMSAATBDataLoader(args,shuffle=True):
    datasets = {
        'train': MMSAATBaselineDataset('train'),
        'valid': MMSAATBaselineDataset('dev'),
        'test': MMSAATBaselineDataset('test')
    }
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=shuffle,
                       collate_fn=pad_collate)
        for ds in datasets.keys()
    }
    
    return dataLoader


labels_en = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
labels_ch = ['愤怒', '厌恶', '恐惧', '高兴', '平静', '悲伤', '惊奇']

class MMSAATBaselineDataset(Dataset):
    def __init__(self, stage):
        
        self.stage = stage
        self.dataset_path = './datasets/CHERMA0723/' + self.stage + '.json'
        
        self.filename_label_list = []

        with open(self.dataset_path) as f:
            for example in json.load(f):
                a = example['audio_file'].replace('.wav', '')
                v = example['video_file']
                self.filename_label_list.append((a, v, example['txt_label'], example['audio_label'], example['visual_label'], example['video_label']))

    def __len__(self):
        return len(self.filename_label_list)

    def __getitem__(self, idx):
        current_filename, current_filename_v, label_t, label_a, label_v, label_m = self.filename_label_list[idx]
        
        text_vector = np.load('./datasets/CHERMA0723/text/' + self.stage + '/' + current_filename + '.npy')
        text_vector = torch.from_numpy(text_vector)

        video_vector = np.load('./datasets/CHERMA0723/visual/' + self.stage + '/' + current_filename + '.mp4.npy')
        video_vector = torch.from_numpy(video_vector)

        audio_vector = np.load('./datasets/CHERMA0723/audio/'+ self.stage + '/' + current_filename + '.npy') 
        audio_vector = torch.from_numpy(audio_vector)

        return  text_vector, audio_vector, video_vector, labels_ch.index(label_t), labels_ch.index(label_a), labels_ch.index(label_v), labels_ch.index(label_m)
