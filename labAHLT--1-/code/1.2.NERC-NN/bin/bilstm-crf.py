
import torch
import torch.nn as nn
import torch.nn.functional as func
import os
from codemaps import Codemaps
from dataset import *
from torch.utils.data import TensorDataset, DataLoader
from evaluator import evaluate
import torch.optim as optim
import random
from TorchCRF import CRF

class nercBiLSTMCRF(nn.Module):
    def __init__(self, codes, embLWsize = 100, embWsize = 100, embSsize = 50, dropout_rate=0.1, lstm_out_size = 200, linear_out_size = 200, num_layers=1, dropout_lstm = False, linear2=False) :
        super(nercBiLSTMCRF, self).__init__()

        n_lc_words = codes.get_n_lc_words()
        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_feat = codes.get_n_features()
        n_labels = codes.get_n_labels()
        self.n_labels = n_labels
        self.dropout_lstm = dropout_lstm
        self.linear2_true = linear2
        self.embLW = nn.Embedding(n_lc_words, embLWsize)
        self.embW = nn.Embedding(n_words, embWsize)
        self.embS = nn.Embedding(n_sufs, embSsize)
        
        self.dropLW = nn.Dropout(dropout_rate)
        self.dropW = nn.Dropout(dropout_rate)
        self.dropS = nn.Dropout(dropout_rate)
        lstm_in_size = embLWsize + embWsize + embSsize + n_feat
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
        
        self.crf = CRF(n_labels)

    def forward(self, lw, w, s, f, tags=None, mask=None):
        x = self.embLW(lw)
        y = self.embW(w)
        z = self.embS(s)
        x = self.dropLW(x)
        y = self.dropW(y)
        z = self.dropS(z)
        f = f.float()
        x = torch.cat((x, y, z, f), dim=2)
        x,_ = self.lstm(x)
        if self.dropout_lstm:
            x = self.dropLSTM(x)        
        x = func.relu(x)
        if self.linear2_true:
            x = self.linear1(x)
            x = func.relu(x)
            x = self.dropFC(x)
            x = self.linear2(x)
            x = func.relu(x)
        else:
            x = self.linear(x)
            x = func.relu(x)
        x = self.out(x)
        if mask is None:
            mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        if tags is not None:
            loss = -self.crf(x, tags, mask=mask).mean()
            return loss
        else:
            pred_tags=self.crf.viterbi_decode(x, mask=mask)
            return pred_tags

used_device = "cuda:0" if torch.cuda.is_available() else "cpu"
def build_mask(w):
    return (w != 0)
def train(network, epoch, train_loader, optimizer):
    network.to(torch.device(used_device))
    network.train()
    seen = 0
    acc_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        X=list(batch)
        target=X.pop()
        lw, w, s, f=X
        mask = build_mask(w)
        optimizer.zero_grad()
        loss = network(lw, w, s, f, tags=target.long(), mask=mask)
        loss.backward()
        optimizer.step()
        acc_loss += loss.item()
        avg_loss = acc_loss / (batch_idx + 1)
        seen += len(w)
    print('Train Epoch {}: batch:{}/{} sentence:{}/{} [{:.2f}%] Loss:{:.6f}\r'.format(
                    epoch,
                    batch_idx+1, len(train_loader),
                    seen, len(train_loader.dataset),
                    100.*(batch_idx+1)/len(train_loader),
                    avg_loss),
            flush=True, end='')
    print()

def validation(network, val_loader):
    network.to(torch.device(used_device))
    network.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            X = list(batch)
            target=X.pop()
            lw, w, s, f=X
            mask = build_mask(w)
            loss = network(lw, w, s, f,tags=target.long(),mask=mask)
            val_loss += loss.item()
            # decode
            pred_seqs=network(lw, w, s, f, mask=mask)
            for i, seq in enumerate(pred_seqs):
                gold=target[i][:len(seq)].tolist()
                for p, g in zip(seq, gold):
                    if p == g:
                        correct+=1
                    total+=1
    val_loss /= len(val_loader)
    acc = 100.0*correct/total if total > 0 else 0.0
    print('Validation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                val_loss,
                correct, total,
                acc))
    return acc
def encode_dataset(ds, codes, batch_size, total_scores=None):
    X = codes.encode_words(ds, total_scores=total_scores)
    if used_device == "cuda:0":
        X = [x.to(torch.device(used_device)) for x in X]
    return DataLoader(TensorDataset(*X), batch_size)

def encode_dataset_labeled(ds, codes, batch_size, total_scores=None):
    X = codes.encode_words(ds, total_scores=total_scores)
    y = codes.encode_labels(ds)
    if used_device == "cuda:0":
        X = [x.to(torch.device(used_device)) for x in X]
        y = y.to(torch.device(used_device))
    return DataLoader(TensorDataset(*X, y), batch_size=batch_size, shuffle=True)

def output_entities(data, preds, codes, outfile) :
    outf = open(outfile, "w")
    for sid,tags in zip(data.sentence_ids(),preds) :
        inside = False
        text,tokens = data.get_sentence_text(sid), data.get_sentence_tokens(sid)
        for k in range(0, min(len(tokens), len(tags), codes.maxlen)):
            y = tags[k]
            tk = tokens[k]
                
            if (y[0]=="B") :
                entity_start = tk.idx
                entity_end = tk.idx + len(tk.text)
                entity_type = y[2:]
                inside = True
            elif (y[0]=="I" and inside) :
                entity_end = tk.idx + len(tk.text)
            elif (y[0]=="O" and inside) :
                print(sid, str(entity_start)+"-"+str(entity_end-1), text[entity_start:entity_end], entity_type, sep="|", file=outf)
                inside = False
        if inside : print(sid, str(entity_start)+"-"+str(entity_end-1), text[entity_start:entity_end], entity_type, sep="|", file=outf)
    outf.close()

def predict(modelname, datafile, outfile, batch_size=16, max_len=150, suf_len=5, step=None, affixes=None, total_scores=None):

    model = torch.load(os.path.join(modelname,"network.nn"),
                    weights_only=False,
                    map_location=torch.device(used_device))   
    model.eval()
    codes = Codemaps(os.path.join(modelname,"codemaps"), maxlen = max_len, suflen=suf_len, step=step, affixes=affixes)

    testdata = Dataset(datafile)
    test_loader = encode_dataset(testdata, codes, batch_size, total_scores=total_scores)

    Y = []
    with torch.no_grad():
        for X in test_loader:
            X=list(X)
            lw, w, s, f=X
            mask=build_mask(w)
            pred_seqs=model(lw, w, s, f, mask=mask)
            for seq in pred_seqs:
                Y.append([codes.idx2label(idx) for idx in seq])

    # extract & evaluate entities with basic model
    output_entities(testdata, Y, codes, outfile)

if __name__ == "__main__":
    random.seed(64)
    torch.manual_seed(64)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(64)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    BINDIR=os.path.abspath(os.path.dirname(__file__)) # location of this file
    NERDIR=os.path.dirname(BINDIR) # one level up
    SOLDIR=os.path.dirname(NERDIR) # one level up
    MAINDIR=os.path.dirname(SOLDIR) # one level up
    DATADIR=os.path.join(MAINDIR,"data") # down to "data"
    UTILDIR=os.path.join(MAINDIR,"util") # down to "util"
    PREPROCESSED_DIR = os.path.join(NERDIR, "preprocessed")
    MODELS_DIR = os.path.join(NERDIR, "models")
    RESULTS_DIR = os.path.join(NERDIR, "results")
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_pck=os.path.join(PREPROCESSED_DIR, "train.pck")
    devel_pck=os.path.join(PREPROCESSED_DIR, "devel.pck")
    test_pck=os.path.join(PREPROCESSED_DIR, "test.pck")
    train_xml=os.path.join(DATADIR, "train.xml")
    devel_xml=os.path.join(DATADIR, "devel.xml")
    test_xml =os.path.join(DATADIR, "test.xml")
    modelname=os.path.join(MODELS_DIR, "bilstm_crf")
    os.makedirs(modelname, exist_ok=True)
    batch_size= 16
    epochs= 10
    lr= 0.001
    embLWsize= 100
    embWsize= 100
    embSsize= 50
    dropout_rate= 0.1
    lstm_out_size= 200
    linear_out_size= 200
    num_layers= 1
    max_len= 150
    suf_len= 5
    dropout_lstm= False
    linear2= False
    traindata=Dataset(train_pck)
    valdata=Dataset(devel_pck)
    codes=Codemaps(traindata, maxlen=max_len, suflen=suf_len)
    train_loader=encode_dataset_labeled(traindata, codes, batch_size)
    val_loader=encode_dataset_labeled(valdata, codes, batch_size)
    network = nercBiLSTMCRF(codes,embLWsize=embLWsize,embWsize=embWsize,embSsize=embSsize,dropout_rate=dropout_rate,lstm_out_size=lstm_out_size,linear_out_size=linear_out_size,num_layers=num_layers,dropout_lstm=dropout_lstm,linear2=linear2)
    network.to(torch.device(used_device))
    codes.save(os.path.join(modelname, "codemaps"))
    optimizer = optim.Adam(network.parameters(), lr=lr)
    best_acc = 0.0
    for epoch in range(1, epochs+1):
        train(network, epoch, train_loader, optimizer)
        acc = validation(network, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(network, os.path.join(modelname, "network.nn"))
            print(f"Best new model saved with accuracy={acc:.2f}%")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    devel_out = os.path.join(RESULTS_DIR, "devel-bilstm_crf.out")
    devel_stats = os.path.join(RESULTS_DIR, "devel-bilstm_crf.stats")
    predict(modelname=modelname,datafile=devel_pck,outfile=devel_out,batch_size=batch_size,max_len=max_len,suf_len=suf_len)
    evaluate("NER", devel_xml, devel_out, devel_stats)
    test_out = os.path.join(RESULTS_DIR, "test-bilstm_crf.out")
    test_stats = os.path.join(RESULTS_DIR, "test-bilstm_crf.stats")
    predict(modelname=modelname,datafile=test_pck,outfile=test_out,batch_size=batch_size,max_len=max_len,suf_len=suf_len)

