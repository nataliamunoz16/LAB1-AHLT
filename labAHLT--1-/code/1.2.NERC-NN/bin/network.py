
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from gensim.models import KeyedVectors
criterion = nn.CrossEntropyLoss()
def load_pretrained_matrix(word2idx, emb_path, emb_dim):

   n = len(word2idx)
   
   rng = np.random.default_rng(2345)
   # random init for all words (covers unknown words)
   matrix = rng.normal(scale=0.1, size=(n, emb_dim)).astype(np.float32)
   # padding index 0 stays as zeros
   matrix[0] = 0.0
   found = 0
   with open(emb_path, encoding="utf-8") as f:
      for line in f:
         parts = line.rstrip().split(" ")
         word  = parts[0]
         if word in word2idx:
            idx = word2idx[word]
            if idx > 0:   # skip padding slot
               matrix[idx] = np.array(parts[1:], dtype=np.float32)
               found += 1
   print(f"Loaded {found} vectors out of {len(word2idx)}")
   return torch.tensor(matrix)

# def load_pretrained_matrix(word2idx, emb_path, emb_dim):
#    n = len(word2idx)
#    # 1. Load the binary model
#    # Note: binary=True is for .bin files
#    model = KeyedVectors.load_word2vec_format(emb_path, binary=True)
   
#    rng = np.random.default_rng(2345)
#    matrix = rng.normal(scale=0.1, size=(n, emb_dim)).astype(np.float32)
#    matrix[0] = 0.0 
   
#    found = 0
#    for word, idx in word2idx.items():
#       if word in model:
#          matrix[idx] = model[word]
#          found += 1
         
#    print(f"Loaded {found} vectors out of {n} from binary file.")
#    return torch.tensor(matrix)
class nercLSTM(nn.Module):
   def __init__(self, codes, embLWsize = 100, embWsize = 100, embSsize = 50, dropout_rate=0.1, lstm_out_size = 200, linear_out_size = 200, num_layers=1, dropout_lstm = False, linear2=False, pretrained_emb_path=None, freeze_pretrained=False, activation='relu', use_lemma=False, use_pref3=False, use_pref5=False, use_pos=False):
      super(nercLSTM, self).__init__()

      n_lc_words = codes.get_n_lc_words()
      n_words = codes.get_n_words()
      n_sufs = codes.get_n_sufs()
      n_feat = codes.get_n_features()
      n_labels = codes.get_n_labels()
      self.use_lemma= use_lemma
      self.use_pref3= use_pref3
      self.use_pref5= use_pref5
      self.use_pos= use_pos
      self.dropout_lstm = dropout_lstm
      self.linear2_true = linear2
      self.embLW = nn.Embedding(n_lc_words, embLWsize)
      #############################################################################################
      self.embW = nn.Embedding(n_words, embWsize)
      if pretrained_emb_path is not None:
         pretrained_matrix = load_pretrained_matrix(word2idx = codes.lc_word_index,emb_path = pretrained_emb_path,emb_dim  = embLWsize)
         # overwrite the default random weights
         self.embLW.weight = nn.Parameter(pretrained_matrix)
         if freeze_pretrained:
            self.embLW.weight.requires_grad = False
      self.embS = nn.Embedding(n_sufs, embSsize)
      #############################################################################################
      self.dropLW = nn.Dropout(dropout_rate)
      self.dropW = nn.Dropout(dropout_rate)
      self.dropS = nn.Dropout(dropout_rate)
      embLsize=100
      embPsize=50
      embPOSsize=50
      
      if self.use_lemma:
         self.embL= nn.Embedding(len(codes.lemma_index), embLsize)
         self.dropL= nn.Dropout(dropout_rate)
      if self.use_pref3:
         self.embP3= nn.Embedding(len(codes.pre3_index), embPsize)
         self.dropP3= nn.Dropout(dropout_rate)
      if self.use_pref5:
         self.embP5= nn.Embedding(len(codes.pre5_index), embPsize)
         self.dropP5= nn.Dropout(dropout_rate)
      if self.use_pos:
         self.embPOS= nn.Embedding(len(codes.pos_index), embPOSsize)
         self.dropPOS= nn.Dropout(dropout_rate)
      
      
      lstm_in_size= embLWsize+embWsize+embSsize+n_feat
      if self.use_lemma:
         lstm_in_size += embLsize
      if self.use_pref3:
         lstm_in_size += embPsize
      if self.use_pref5:
         lstm_in_size += embPsize
      if self.use_pos:
         lstm_in_size += embPOSsize
      
      # self.dropL = nn.Dropout(dropout_rate)
      # self.dropP = nn.Dropout(dropout_rate)
      # self.dropPOS = nn.Dropout(dropout_rate)
      
      self.lstm = nn.LSTM(lstm_in_size, lstm_out_size, num_layers = num_layers, bidirectional=True, batch_first=True)
      if self.dropout_lstm:
         self.dropLSTM = nn.Dropout(dropout_rate)
      if self.linear2_true:
         self.linear1 = nn.Linear(2*lstm_out_size, linear_out_size)
         self.dropFC = nn.Dropout(dropout_rate)
         self.linear2 = nn.Linear(linear_out_size, linear_out_size // 2)
         self.out = nn.Linear(linear_out_size//2, n_labels)
      else:
         self.linear = nn.Linear(2*lstm_out_size, linear_out_size)
         self.out = nn.Linear(linear_out_size, n_labels)
      activations = {'relu':func.relu,'tanh':torch.tanh,'gelu':func.gelu,'leaky':func.leaky_relu,'elu':func.elu}
      self.act = activations[activation]

   # def forward(self, lw, w, s, f,l,p, pos):
   #    x = self.embLW(lw)
   #    y = self.embW(w)
   #    z = self.embS(s)
      
   #    x_l  = self.act(self.embL(l))
   #    x_p  = self.act(self.embP(p))
   #    x_pos = self.embPOS(pos)
      
      
   #    x = self.dropLW(x)
   #    y = self.dropW(y)
   #    z = self.dropS(z)
   #    x_l = self.dropL(x_l)
   #    x_p = self.dropP(x_p)
   #    x_pos = self.dropPOS(x_pos)
      
   #    f = f.float()
      
   #    x = torch.cat((x, y, z, f, x_l, x_p, x_pos), dim=2)
   #    x = self.lstm(x)[0]
   #    if self.dropout_lstm:
   #       x = self.dropLSTM(x)        
   #    x = self.act(x)
   #    if self.linear2_true:
   #       x = self.linear1(x)
   #       x = self.act(x)
   #       x = self.dropFC(x)
   #       x = self.linear2(x)
   #       x = self.act(x)
   #    else:
   #       x = self.linear(x)
   #    x = self.out(x)
   #    return x
   def forward(self, lw, w, s, f, l, p3, p5, pos):
      x_lw= self.dropLW(self.embLW(lw))
      x_w= self.dropW(self.embW(w))
      x_s= self.dropS(self.embS(s))
      f= f.float()
      pieces= [x_lw, x_w, x_s, f]
      if self.use_lemma:
         x_l= self.dropL(self.act(self.embL(l)))
         pieces.append(x_l)
      if self.use_pref3:
         x_p3= self.dropP3(self.act(self.embP3(p3)))
         pieces.append(x_p3)
      if self.use_pref5:
         x_p5= self.dropP5(self.act(self.embP5(p5)))
         pieces.append(x_p5)
      if self.use_pos:
         x_pos = self.dropPOS(self.embPOS(pos))
         pieces.append(x_pos)
      x = torch.cat(pieces, dim=2)
      x = self.lstm(x)[0]
      if self.dropout_lstm:
         x = self.dropLSTM(x)
      x = self.act(x)
      if self.linear2_true:
         x = self.linear1(x)
         x = self.act(x)
         x = self.dropFC(x)
         x = self.linear2(x)
         x = self.act(x)
      else:
         x = self.linear(x)
      x = self.out(x)
      return x
   


