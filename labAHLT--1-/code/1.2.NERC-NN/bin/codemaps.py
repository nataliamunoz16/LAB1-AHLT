import os
import string
import re
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from dataset import *
from functions_for_extraction import intervals_cosine

# folder where this file is located
THISDIR=os.path.abspath(os.path.dirname(__file__))
# go two folders up and locate "resources" folder there
NERDIR=os.path.dirname(THISDIR)
SOLDIR=os.path.dirname(NERDIR)
MAINDIR=os.path.dirname(SOLDIR)
RESOURCESDIR=os.path.join(MAINDIR, "resources")

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen = 150, suflen=5,preflen=3,step=None, affixes=None) :
        # maxlen = params['max_len'] if 'max_len' in params else None
        # suflen = params['suf_len'] if 'suf_len' in params else None
        
        #----------------------
        self.step = step or []
        self.affixes = affixes or {}
        self.external = {}
        self.externalpart = {}
        with open(os.path.join(RESOURCESDIR,"HSDB.txt"),encoding='utf-8') as h :
            for x in h.readlines() :
                x = x.strip().lower()
                self.external[x] = {"any"}
                wds = x.split()
                if len(wds)>1 :
                   for w in wds:
                       self.externalpart[w] = {"any"}
                                
        with open(os.path.join(RESOURCESDIR,"DrugBank.txt"),encoding='utf-8') as h :
            for x in h.readlines() :
                (n,t) = x.strip().lower().split("|")
                if n in self.external : self.external[n].add(t)
                else: self.external[n] = {t}
                wds = n.split()
                if len(wds)>1 :
                   for w in wds:
                       if w in self.externalpart :
                          self.externalpart[w].add(t)
                       else :
                          self.externalpart[w] = {t}
                                
        #----------------------
                
        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen, preflen)

        elif type(data) == str :
            print('Codemaps: ', end='')
            if maxlen is not None or suflen is not None :
                print('Ignoring given params and ', end='')
            print(f'loading index from {data}.idx')
            self.__load(data)

        else:
            print(f'codemaps: Missing max_len and/or suf_len parameters')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen, preflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        words = set([])
        lc_words = set([])
        sufs = set([])
        labels = set([])
        
        lemmas = set([])
        prefs3 = set([])
        prefs5 = set([])
        pos_tags = set([])
        self.preflen = preflen
        
        for _,tokens,lab in data.sentences() :
            for i,t in enumerate(tokens) :
                if t.text.startswith(" "): continue
                words.add(t.text)
                lc_words.add(t.text.lower())
                sufs.add(t.text.lower()[-self.suflen:])
                labels.add(lab[i])
                lemmas.add(t.lemma_.lower()) 
                prefs3.add(t.text.lower()[:3])
                prefs5.add(t.text.lower()[:5])
                pos_tags.add(t.tag_)

        self.word_index = {w: i+2 for i,w in enumerate(sorted(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.lc_word_index = {w: i+2 for i,w in enumerate(sorted(lc_words))}
        self.lc_word_index['PAD'] = 0 # Padding
        self.lc_word_index['UNK'] = 1 # Unknown words
        
        self.suf_index = {s: i+2 for i,s in enumerate(sorted(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        self.label_index = {t: i+1 for i,t in enumerate(sorted(labels))}
        self.label_index['PAD'] = 0 # Padding
        
        self.lemma_index = {w: i+2 for i, w in enumerate(sorted(lemmas))}
        self.lemma_index['PAD'], self.lemma_index['UNK'] = 0, 1

        # self.pre_index = {p: i+2 for i, p in enumerate(sorted(prefs))}
        # self.pre_index['PAD'], self.pre_index['UNK'] = 0, 1
        self.pre3_index = {p: i+2 for i, p in enumerate(sorted(prefs3))}
        self.pre3_index['PAD'], self.pre3_index['UNK'] = 0, 1

        self.pre5_index = {p: i+2 for i, p in enumerate(sorted(prefs5))}
        self.pre5_index['PAD'], self.pre5_index['UNK'] = 0, 1
        
        self.pos_index = {tag: i+2 for i, tag in enumerate(sorted(list(pos_tags)))}
        self.pos_index['PAD'] = 0
        self.pos_index['UNK'] = 1
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.lc_word_index = {}
        self.suf_index = {}
        self.label_index = {}
        self.lemma_index = {} 
        # self.pre_index = {}   
        self.pre3_index = {}
        self.pre5_index = {}
        self.pos_index = {}

        with open(name+".idx") as f :
            for line in f.readlines(): 
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'LCWORD': self.lc_word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                elif t == 'LEMMA': self.lemma_index[k] = int(i)
                # elif t == 'PREFIX': self.pre_index[k] = int(i)
                elif t == 'PREFIX3': self.pre3_index[k] = int(i)
                elif t == 'PREFIX5': self.pre5_index[k] = int(i)
                elif t == 'POS': self.pos_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            print ('PREFLEN', self.preflen, "-", file=f)
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.lc_word_index : print('LCWORD', key, self.lc_word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            for key in self.lemma_index : print('LEMMA', key, self.lemma_index[key], file=f)
            for key in self.pre3_index : print('PREFIX3', key, self.pre3_index[key], file=f)
            for key in self.pre5_index : print('PREFIX5', key, self.pre5_index[key], file=f)
            for key in self.pos_index : print('POS', key, self.pos_index[key], file=f)


    ## --------- Pad tensors for short sentences and cut sentences longer 
    ## --------- than maxlen, so all sentences have the same length.
    ## --------- Return a tensor with all the sentences.
    ## --------- Given tensor_list is assumed to have one tensor per sentence.
    ## --------- Each sentence tensors has :
    ## ---------    1nd dimension = n_words in the sentence
    ## ---------    2nd dimension (if any) = n_feature bits for each word
    def cut_and_pad(self, tensor_list, pad) :
        # check if the tensors are 1d or 2d, and decide shape of output tensor 
        if len(tensor_list[0].shape)==1 : 
            shape = (len(tensor_list), self.maxlen)
        elif len(tensor_list[0].shape)==2 : 
            shape = (len(tensor_list), self.maxlen, tensor_list[0].shape[1])
        # cut sentences longer than maxlen
        tensor_list = [s[0:self.maxlen] for s in tensor_list]
        # create a tensor full of padding with the final desired shape
        padded = torch.Tensor([]).new_full(shape, pad, dtype=torch.int64)        
        # fill padded tensor with given data, leaving padding in unused spaces
        for i,s in enumerate(tensor_list):
            for j,f in enumerate(tensor_list[i]) :
                padded[i,j] = f
        return padded
    
    ## --------- encode X from given data ----------- 
    def encode_words(self, data, total_scores=None) :

        #----- encode sentence words
        enc = [torch.Tensor([self.word_index[w.text] if w.text in self.word_index else self.word_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xw = self.cut_and_pad(enc, self.word_index['PAD'])

        #------ encode sentence lowercase words
        enc = [torch.Tensor([self.lc_word_index[w.text.lower()] if w.text.lower() in self.lc_word_index else self.lc_word_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xlw = self.cut_and_pad(enc, self.lc_word_index['PAD'])
        
        #------ encode sentence suffixes
        enc = [torch.Tensor([self.suf_index[w.text.lower()[-self.suflen:]] if w.text.lower()[-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xs = self.cut_and_pad(enc, self.suf_index['PAD'])
        
        # Encode lemmas
        enc_l = [torch.Tensor([self.lemma_index[w.lemma_.lower()] if w.lemma_.lower() in self.lemma_index else self.lemma_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xl = self.cut_and_pad(enc_l, self.lemma_index['PAD'])

        # Encode prefixes
        enc_p3 = [torch.Tensor([self.pre3_index[w.text.lower()[:3]] if w.text.lower()[:3] in self.pre3_index else self.pre3_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xp3 = self.cut_and_pad(enc_p3, self.pre3_index['PAD'])

        enc_p5 = [torch.Tensor([self.pre5_index[w.text.lower()[:5]] if w.text.lower()[:5] in self.pre5_index else self.pre5_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xp5 = self.cut_and_pad(enc_p5, self.pre5_index['PAD'])
        
        # Encode postags
        enc_pos = [torch.Tensor([self.pos_index[w.tag_] if w.tag_ in self.pos_index else self.pos_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xpos = self.cut_and_pad(enc_pos, self.pos_index['PAD'])

        #------ encode word features
        # enc = [torch.Tensor([self.features(w) for w in s]) for _,s,_ in data.sentences()]
        enc_feat = []
        for jj, (_, s, _) in enumerate(data.sentences()):
            sent_feat = []
            for ii, w in enumerate(s):
                sent_feat.append(self.features(w, total_scores=total_scores, jj=jj, ii=ii))
            enc_feat.append(torch.Tensor(sent_feat))
        Xf = self.cut_and_pad(enc_feat, 0)
        
        

        # return encoded sequences
        print(f"Xlw:{Xlw.shape},Xw:{Xw.shape},Xs:{Xs.shape},Xf:{Xf.shape},Xl:{Xl.shape},Xp3:{Xp3.shape},Xp5:{Xp5.shape}, Xpos:{Xpos.shape}")
        return [Xlw, Xw, Xs, Xf, Xl, Xp3, Xp5, Xpos]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels
        enc = [torch.Tensor([self.label_index[lab] for lab in l]) for _,_,l in data.sentences()]
        Y = self.cut_and_pad(enc, self.label_index['PAD'])
        return Y

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get lc_word index size ---------
    def get_n_lc_words(self) :
        return len(self.lc_word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    def get_n_lemmas(self):
        return len(self.lemma_index)

    # def get_n_prefs(self):
    #     return len(self.pre_index)

    def get_n_pos(self):
        return len(self.pos_index)
    ## -------- get label index size ---------
    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given lc_word ---------
    def lcword2idx(self, w) :
        return self.lc_word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

    def extra_features(self, w=None, total_scores=None, jj=None, ii=None):
        form = w.text
        lcform = w.text.lower()
        f_crf = []
        if len(self.step)>0:
                if 0 in self.step:
                    f_ = [0]
                    #isStopword
                    if w.lower_ in STOP_WORDS: f_[0] = 1
                    f_crf = f_crf + f_
                if 1 in self.step:
                    f_ = [0] * 4
                    #TopSuf
                    n=3
                    tlow = w.text.lower()
                    suf  = tlow[-n:]
                    for i, ent_type in enumerate(["drug", "group", "brand", "drug_n"]):
                        if suf in self.affixes.get(f"{ent_type}.suf{n}", []):
                            f_[i]=1
                    f_crf = f_crf + f_
                if 2 in self.step:
                    f_ = [0]
                    #hasInternalCap
                    if re.search(r'[A-Z]', form[1:]): f_[0] = 1
                    f_crf = f_crf + f_
                if 3 in self.step:
                    f_ = [0]
                    #hasSlash
                    if "/" in lcform: f_[0] = 1
                    f_crf = f_crf + f_
                if 4 in self.step:
                    f_ = [0]
                    #hasPlus
                    if "+" in lcform: f_[0] = 1
                    f_crf = f_crf + f_
                if 5 in self.step:
                    f_ = [0]
                    #hasMinus
                    if "-" in lcform: f_[0] = 1
                    f_crf = f_crf + f_
                if 6 in self.step:
                    #lenToken
                    f_ = [0] * 5
                    L = len(lcform)
                    index=(0 if L==1 else
                        1 if L<=3 else
                        2 if L<=6 else
                        3 if L<=10 else
                        4)
                    f_[index] = 1
                    f_crf = f_crf + f_
                if 7 in self.step:
                    f_ = [0]
                    #isTrigger
                    triggers = {"dose","doses","therapy","treatment","treated","administered","administration","antibiotic","antagonist"}
                    if lcform in triggers: f_[0] = 1
                    f_crf = f_crf + f_
                if 8 in self.step:
                    f_ = [0]*28
                    for k, key_ in enumerate(["drug", "group", "brand", "drug_n"]):
                        value_cosine = intervals_cosine(total_scores[key_][jj][ii])
                        f_[value_cosine+7*k]=1
                    f_crf = f_crf + f_
        return f_crf
    def get_n_features(self):
        n = 16
        sizes = {0: 1,1: 4,2: 1,3: 1,4: 1,5: 1,6: 5,7: 1,8: 28,}
        for s in self.step:
            n += sizes[s]
        return n
    ## -------- create vector with binary features (used by encode_words)
    def features(self, w, total_scores=None, jj=None, ii=None):
        f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if w is not None :
            form = w.text
            if form.isupper(): f[0] = 1
            if form.istitle(): f[1] = 1
            if form.isdigit(): f[2] = 1
            if '-' in form:    f[3] = 1
            if re.search('[0-9]',form): f[4] = 1
            if any([c in string.punctuation for c in form]): f[5] = 1

            lcform = w.text.lower()
            if lcform in self.external :
                if 'drug' in self.external[lcform] : f[6] = 1
                if 'group' in self.external[lcform] : f[7] = 1
                if 'brand' in self.external[lcform] : f[8] = 1
                if 'drug_n' in self.external[lcform] : f[9] = 1
                if 'any' in self.external[lcform] : f[10] = 1
            if lcform in self.externalpart :
                if 'drug' in self.externalpart[lcform] : f[11] = 1
                if 'group' in self.externalpart[lcform] : f[12] = 1
                if 'brand' in self.externalpart[lcform] : f[13] = 1
                if 'drug_n' in self.externalpart[lcform] : f[14] = 1
                if 'any' in self.externalpart[lcform] : f[15] = 1
            extra = self.extra_features(w=w, total_scores=total_scores, jj=jj, ii=ii)
            f = f + extra

        return f





