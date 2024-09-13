'''
* @name: almt.py
* @description: Implementation of ALMT
'''

import torch
from torch import nn
from core.utils import JS_divergence
from .LSTM_encoder import RNNEncoder
from transformers import BertModel
import torch.nn.functional as F
device = torch.device("cuda")

class FusionNet(nn.Module):

    def __init__(self, in_size, hidden_size, dropout):

        super(FusionNet, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        return y_2

class TISL(nn.Module):
    def __init__(self, args, bert_pretrained='bert-base-chinese'):
        super(TISL, self).__init__()
        self.args = args
        inner_feature_dim = 768
        
        self.bertmodel = BertModel.from_pretrained(bert_pretrained)
        self.text_f_encoder = RNNEncoder(in_size = args.feature_dims[0], hidden_size = inner_feature_dim, out_size = inner_feature_dim, num_layers = args.LSTM_layers, dropout = args.dropout, bidirectional = False, fs = True)
        self.text_c_encoder = RNNEncoder(in_size = args.feature_dims[0], hidden_size = inner_feature_dim, out_size = inner_feature_dim, num_layers = args.LSTM_layers, dropout = args.dropout, bidirectional = False, fs = True)
        self.acoustic_f_encoder = RNNEncoder(in_size = args.feature_dims[1], hidden_size = inner_feature_dim, out_size = inner_feature_dim, num_layers = args.LSTM_layers, dropout = args.dropout, bidirectional = False, fs = True)
        self.acoustic_c_encoder = RNNEncoder(in_size = args.feature_dims[1], hidden_size = inner_feature_dim, out_size = inner_feature_dim, num_layers = args.LSTM_layers, dropout = args.dropout, bidirectional = False, fs = True)
        self.visual_f_encoder = RNNEncoder(in_size = args.feature_dims[2], hidden_size = inner_feature_dim, out_size = inner_feature_dim, num_layers = args.LSTM_layers, dropout = args.dropout, bidirectional = False, fs = True)
        self.visual_c_encoder = RNNEncoder(in_size = args.feature_dims[2], hidden_size = inner_feature_dim, out_size = inner_feature_dim, num_layers = args.LSTM_layers, dropout = args.dropout, bidirectional = False, fs = True)
        self.fusion = FusionNet(in_size=3*inner_feature_dim,hidden_size=inner_feature_dim,dropout=args.dropout)

        self.cls_head_M = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        self.cls_head_T_fuse = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        self.cls_head_T_uni = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        self.cls_head_A_fuse = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        self.cls_head_A_uni = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        self.cls_head_V_fuse = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        self.cls_head_V_uni = nn.Sequential(nn.Linear(inner_feature_dim, 256),nn.ReLU(),nn.Linear(256, 64),nn.ReLU(),nn.Linear(64, 1))
        
    def forward(self, text, audio, visual, vlens, alens):

        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        text = self.bertmodel(input_ids = input_ids, attention_mask = input_mask ,token_type_ids = segment_ids)[0]
        
        tlens = torch.Tensor(text.size()[0]).fill_(self.args.seq_lens[0])
        text_f = self.text_f_encoder(text,tlens)
        text_c = self.text_c_encoder(text,tlens)
        Djs_T = JS_divergence(text_f, text_c)

        audio_f = self.acoustic_f_encoder(audio, alens)
        audio_c = self.acoustic_c_encoder(audio, alens)
        Djs_A = JS_divergence(audio_f, audio_c)

        visual_f = self.visual_f_encoder(visual, vlens)
        visual_c = self.visual_c_encoder(visual, vlens)
        Djs_V = JS_divergence(visual_f, visual_c)
        
        fusion = torch.cat((torch.cat((text_f,audio_f),dim=1),visual_f),dim=1)
        fusion_out = self.fusion(fusion)


        output_M = self.cls_head_M(fusion_out)
        output_T_f = self.cls_head_T_fuse(text_f)
        output_T_c = self.cls_head_T_uni(text_c)
        output_A_f = self.cls_head_A_fuse(audio_f)
        output_A_c = self.cls_head_A_uni(audio_c)
        output_V_f = self.cls_head_V_fuse(visual_f)
        output_V_c = self.cls_head_V_uni(visual_c)

        return output_M, output_T_f, output_T_c, output_A_f, output_A_c, output_V_f, output_V_c, Djs_T, Djs_A, Djs_V

def build_model(opt):
    model = TISL(opt, bert_pretrained = 'bert-base-chinese')
    return model

