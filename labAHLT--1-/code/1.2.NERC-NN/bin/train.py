#! /usr/bin/python3

import sys, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import shutil
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary
from predict import predict
from evaluator import evaluate

from dataset import *
from codemaps import *

from network import nercLSTM, criterion
SEED=64
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
   torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def set_seed(seed=SEED):
   os.environ["PYTHONHASHSEED"] = str(seed)
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True)

# use gpu if available
used_device = "cuda:0" if torch.cuda.is_available() else "cpu"

#----------------------------------------------
def train(network, epoch, train_loader, optimizer):
   network.to(torch.device(used_device))

   network.train()
   seen = 0
   acc_loss = 0
   for batch_idx, batch in enumerate(train_loader):
      X = list(batch)
      target = X.pop()
      optimizer.zero_grad()
      output = network(*X)
      output = output.flatten(0,1)
      target = target.flatten(0,1)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      acc_loss += loss.item()
      avg_loss = acc_loss/(batch_idx+1)
      seen += len(X[0])
   print('Train Epoch {}: batch:{}/{} sentence:{}/{} [{:.2f}%] Loss:{:.6f}\r'.format(
                  epoch,
                  batch_idx+1, len(train_loader),
                  seen, len(train_loader.dataset),
                  100.*(batch_idx+1)/len(train_loader),
                  avg_loss),
         flush=True, end='')
   print()

#----------------------------------------------
def validation(network, val_loader):
   network.eval()
   test_loss = 0
   correct = 0
   total = 0
   with torch.no_grad():
      for batch in val_loader:
         X = list(batch)
         target = X.pop()
         output = network(*X)
         output = output.flatten(0,1)
         target = target.flatten(0,1)
         test_loss += criterion(output, target).item()
         pred = output.data.max(1, keepdim=True)[1]
         correct += pred.eq(target.data.view_as(pred)).sum()
         total += target.size()[0]
   test_loss /= len(val_loader)
   acc = 100.*correct/total
   print('Validation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss,
            correct, total,
            acc))
   return acc

#----------------------------------------------
def encode_dataset(ds, codes, batch_size, total_scores=None, seed=2345):
   X = codes.encode_words(ds, total_scores=total_scores)
   y = codes.encode_labels(ds)
   if used_device == "cuda:0":
      X = [x.to(torch.device(used_device)) for x in X]
      y = y.to(torch.device(used_device))
   g = torch.Generator()
   g.manual_seed(seed)
   return DataLoader(TensorDataset(*X, y), batch_size=batch_size, generator=g)


#----------------------------------------------
def do_train(trainfile, 
            valfile, 
            modelname, 
            epochs = 10,
            batch_size = 16,
            optimizer = 'Adam',
            lr = 0.001,
            embLWsize = 100,
            embWsize = 100,
            embSsize = 50,
            dropout_rate=0.1,
            lstm_out_size = 200,
            linear_out_size = 200,
            num_layers_lstm=1,
            max_len = 150,
            suf_len=5,
            pref_len=3,
            dropout_lstm = False,
            linear2=False,
            step=None,
            affixes=None,
            total_scores_train=None,
            total_scores_val=None,
            seed=2345,
            pretrained_emb_path=None, 
            freeze_pretrained=False, 
            activation='relu',
            use_lemma=False,
            use_pref3=False,
            use_pref5=False,
            use_pos=False):

   # set default values if some parameter is missing       
   #  if 'max_len' not in params : params['max_len'] = 150
   #  if 'suf_len' not in params : params['suf_len'] = 5
   #  if 'batch_size' not in params : params['batch_size'] = 16
   #  if 'epochs' not in params : params['epochs'] = 10
   
   set_seed(seed)
   # load pickle datasets (or parse if needed)
   traindata = Dataset(trainfile)
   valdata = Dataset(valfile)

   # create indexes from training data
   codes  = Codemaps(traindata, maxlen = max_len, suflen=suf_len,preflen=pref_len, step=step,affixes=affixes)
   # encode datasets
   train_loader = encode_dataset(traindata, codes, batch_size, total_scores=total_scores_train)
   val_loader = encode_dataset(valdata, codes, batch_size, total_scores=total_scores_val)

   # build network
   network =  nercLSTM(codes, embLWsize = embLWsize, embWsize = embWsize, embSsize = embSsize, dropout_rate=dropout_rate, lstm_out_size = lstm_out_size, linear_out_size = linear_out_size, num_layers=num_layers_lstm, dropout_lstm = dropout_lstm, linear2=linear2,pretrained_emb_path=pretrained_emb_path, freeze_pretrained=freeze_pretrained, activation=activation, use_lemma=use_lemma,use_pref3=use_pref3,use_pref5=use_pref5,use_pos=use_pos)

   summary(network)

   # save indexs
   os.makedirs(modelname,exist_ok=True)
   torch.save(network, os.path.join(modelname,"network.nn"))
   codes.save(os.path.join(modelname,"codemaps"))
   # train each epoch, keep the best model on validation
   best = 0 
   if optimizer == 'Adam':
      optimizer_selected = optim.Adam(network.parameters(), lr=lr)
   elif optimizer == "AdamW":
      optimizer_selected= optim.AdamW(network.parameters(), lr=lr)
   else:
      raise ValueError(f"Optimizer not supported: {optimizer}")
         
   for epoch in range(epochs):
      train(network, epoch, train_loader, optimizer=optimizer_selected)
      acc = validation(network, val_loader)
      if acc>best :
         best = acc
         torch.save(network, os.path.join(modelname,f"network.nn"))

def generate_experiments(params):
   default = {k: v[0] for k, v in params.items()}
   experiments = []
   experiments.append(default.copy())
   for key, values in params.items():
      for val in values[1:]: 
         exp = default.copy()
         exp[key] = val
         experiments.append(exp)
   return experiments

def build_name(params):
   parts = []
   for k, v in params.items():
      parts.append(f"{k}{v}")
   return "_".join(parts)

def train_with_different_hyperparameters():
   hyperparams= {
      # "epochs": [10],
      # "batch_size" : [16, 32, 64],
      # "optimizer" : ['Adam', 'AdamW'],
      # "lr" : [0.001, 0.01, 0.00001],
      "embLWsize" : [100], #"embLWsize" : [100, 50, 200],
      # "embWsize" : [100, 50, 200],
      # "embSsize" : [50, 80, 25],
      # "dropout_rate": [0.1, 0.2, 0.4],
      # "lstm_out_size" : [200, 100, 300],
      # "linear_out_size" : [200, 100, 300],
      # "num_layers_lstm": [1, 2, 3],
      # "max_len" : [150],
      # "suf_len":[5, 2, 3],
      # "dropout_lstm" : [False, True],
      # "linear2":[False, True],
      # "activation":['relu','tanh','gelu','leaky','elu'],
      # "pretrained_emb_path": ["C:/Users/Natalia/Desktop/MAI/2N QUATRI/AHLT/labAHLT (1)/code/1.2.NERC-NN/bin/PubMed-and-PMC-w2v.bin", "C:/Users/Natalia/Desktop/MAI/2N QUATRI/AHLT/labAHLT (1)/code/1.2.NERC-NN/bin/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"]
      "pretrained_emb_path": ["C:/Users/Natalia/Desktop/MAI/2N QUATRI/AHLT/labAHLT (1)/code/1.2.NERC-NN/bin/wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"],
      }
   experiments = generate_experiments(hyperparams)
   default_path = "C:/Users/Rafael/OneDrive/MAI/labAHLT--1-/code/1.2.NERC-NN"
   datadir = "C:/Users/Rafael/OneDrive/MAI/labAHLT--1-/data"

   # Create the list of experiments (Base + the variations)

   os.makedirs(os.path.join(default_path,"models"), exist_ok=True)
   for i, experiment in enumerate(experiments):
      name = build_name(experiment).replace(":","").replace("/","_").replace("\\","_")
      # train model
      print(f"Experiment {i+1}/{len(experiments)}")
      print(f"Training model {name} ...")
      do_train(os.path.join(default_path, "preprocessed","train.pck"), 
               os.path.join(default_path, "preprocessed","devel.pck"), 
               os.path.join(default_path,"models",name), 
               # epochs = experiment['epochs'],
               # batch_size = experiment["batch_size"],
               # optimizer = experiment["optimizer"],
               # lr = experiment["lr"],
               embLWsize = experiment["embLWsize"],
               # embWsize = experiment["embWsize"],
               # embSsize = experiment["embSsize"],
               # dropout_rate=experiment["dropout_rate"],
               # lstm_out_size = experiment["lstm_out_size"],
               # linear_out_size = experiment["linear_out_size"],
               # num_layers_lstm=experiment["num_layers_lstm"],
               # max_len = experiment["max_len"],
               # suf_len=experiment["suf_len"],
               # dropout_lstm = experiment["dropout_lstm"],
               # linear2=experiment["linear2"],
               pretrained_emb_path = experiment["pretrained_emb_path"],
               # activation = experiment["activation"]
               )
      # do_train(os.path.join(default_path, "preprocessed","train.pck"),
      #       os.path.join(default_path, "preprocessed","devel.pck"),
      #       params,
      #       os.path.join(default_path,"models",name))    
      os.makedirs(os.path.join(default_path,"results"), exist_ok=True)
      # run model on test data and evaluate results
      print(f"Running {name} model on test...")
      predict(os.path.join(default_path,"models",name),
               os.path.join(default_path, "preprocessed","test.pck"),
               os.path.join(default_path,"results","test-"+name+".out"),
               # batch_size = experiment['batch_size'],
               # max_len = experiment["max_len"],
               # suf_len=experiment["suf_len"]
               )
      evaluate("NER", os.path.join(datadir,"test.xml"),
               os.path.join(default_path,"results","test-"+name+".out"),
               os.path.join(default_path,"results","test-"+name+".stats"))
      # run model on devel data and evaluate results
      print(f"Running {name} model on devel...")
      predict(os.path.join(default_path,"models",name),
               os.path.join(default_path, "preprocessed","devel.pck"),
               os.path.join(default_path,"results","devel-"+name+".out"),
               # batch_size = experiment['batch_size'],
               # max_len = experiment["max_len"],
               # suf_len=experiment["suf_len"]
               )
      evaluate("NER", os.path.join(datadir,"devel.xml"),
               os.path.join(default_path,"results","devel-"+name+".out"),
               os.path.join(default_path,"results","devel-"+name+".stats"))
def read_f1_from_stats(statsfile):
   """
   Reads the F1 score from the evaluator .stats file
   """
   with open(statsfile, "r", encoding="utf-8") as f:
      for line in f:
         if line.startswith("M.avg"):
               extracted=line.split()
               # last column is F1
               f1=extracted[-1].replace("%", "")
               return float(f1)
   raise ValueError(f"M.avg was not found in {statsfile}")
def run_single_experiment(experiment, model_path, devel_out_path, devel_stats_path, default_path, datadir):
   os.makedirs(os.path.join(default_path, "models"), exist_ok=True)
   os.makedirs(os.path.join(default_path, "results"), exist_ok=True)
   do_train(os.path.join(default_path, "preprocessed","train.pck"), 
            os.path.join(default_path, "preprocessed","devel.pck"), 
            model_path, 
            epochs = experiment['epochs'],
            batch_size = experiment["batch_size"],
            optimizer = experiment["optimizer"],
            lr = experiment["lr"],
            embLWsize = experiment["embLWsize"],
            embWsize = experiment["embWsize"],
            embSsize = experiment["embSsize"],
            dropout_rate=experiment["dropout_rate"],
            lstm_out_size = experiment["lstm_out_size"],
            linear_out_size = experiment["linear_out_size"],
            num_layers_lstm=experiment["num_layers_lstm"],
            max_len = experiment["max_len"],
            suf_len=experiment["suf_len"],
            pref_len=experiment["pref_len"],
            dropout_lstm = experiment["dropout_lstm"],
            linear2=experiment["linear2"],
            activation = experiment["activation"],
            pretrained_emb_path=experiment["pretrained_emb_path"])
   predict(model_path,
         os.path.join(default_path, "preprocessed","devel.pck"),
         devel_out_path,
         batch_size = experiment['batch_size'],
         max_len = experiment["max_len"],
         suf_len=experiment["suf_len"],
         pref_len=experiment["pref_len"])
   evaluate("NER", os.path.join(datadir,"devel.xml"),
            devel_out_path,
            devel_stats_path)
   f1 = read_f1_from_stats(devel_stats_path)
   print(f"Devel f1 = {f1:.4f}")
   return f1

def copy_temporal_paths(temp_model_path, temp_out_path, temp_stats_path, best_model_path, best_out_path, best_stats_path):
   if os.path.isdir(temp_model_path):
      if os.path.exists(best_model_path):
         shutil.rmtree(best_model_path)
      shutil.copytree(temp_model_path, best_model_path)
   else:
      raise FileNotFoundError(f"Temporary path not found: {temp_model_path}")
   shutil.copy2(temp_out_path, best_out_path)
   shutil.copy2(temp_stats_path, best_stats_path)

def greedy_best_hyperparameters():
   hyperparams= {"epochs": [10],
      "batch_size" : [16],
      "optimizer" : ['Adam', 'AdamW'],
      "lr" : [0.001, 0.01],
      "embLWsize" : [100],
      "embWsize" : [100, 50],
      "embSsize" : [50, 25],
      "dropout_rate": [0.1, 0.2],
      "lstm_out_size" : [200, 300],
      "linear_out_size" : [200, 300],
      "num_layers_lstm": [1],
      "max_len" : [150],
      "suf_len":[5, 2],
      "pref_len":[2,3,4],
      "dropout_lstm" : [False, True],
      "linear2":[False, True],
      "activation":["relu", "elu"],
      "pretrained_emb_path": [None]}
   default_path = "C:/Users/Natalia/Desktop/MAI/2N QUATRI/AHLT/labAHLT (1)/code/1.2.NERC-NN"
   datadir = "C:/Users/Natalia/Desktop/MAI/2N QUATRI/AHLT/labAHLT (1)/data"
   models_dir =os.path.join(default_path, "models")
   results_dir =os.path.join(default_path, "results")
   os.makedirs(models_dir, exist_ok=True)
   os.makedirs(results_dir, exist_ok=True)

   temp_model_path = os.path.join(models_dir, "temp_greedy_model")
   temp_devel_out = os.path.join(results_dir, "temp_greedy_devel.out")
   temp_devel_stats = os.path.join(results_dir, "temp_greedy_devel.stats")

   current_best = {k:v[0] for k,v in hyperparams.items()}
   remaining=[k for k,v in hyperparams.items() if len(v)>1]
   print("Evaluating default configuration")
   best_name ="best_" + build_name(current_best).replace("/", "-").replace("\\", "-").replace("pretrained_emb_path", "pre")
   best_name = f"{best_name}_{SEED}"
   best_score = run_single_experiment(current_best, temp_model_path, temp_devel_out, temp_devel_stats, default_path, datadir)
   best_model_path = os.path.join(models_dir, best_name)
   best_devel_out = os.path.join(results_dir, f"devel-{best_name}.out")
   best_devel_stats = os.path.join(results_dir, f"devel-{best_name}.stats")
   copy_temporal_paths(temp_model_path, temp_devel_out, temp_devel_stats, best_model_path, best_devel_out, best_devel_stats)
   
   improved = True
   while remaining and improved:
      print("Running new cycle")
      improved=False
      round_best_score=best_score
      round_best_config=None
      round_best_key=None
      for key in remaining:
         values=hyperparams[key][1:]
         for val in values:
            candidate = current_best.copy()
            candidate[key]=val
            if key== "pretrained_emb_path":
               candidate["embLWsize"] = 200
            print(f"Trying change: {key}: {val}")
            if os.path.exists(temp_model_path):
               shutil.rmtree(temp_model_path)
            if os.path.exists(temp_devel_out):
               os.remove(temp_devel_out)
            if os.path.exists(temp_devel_stats):
               os.remove(temp_devel_stats)
            score = run_single_experiment(candidate, temp_model_path, temp_devel_out, temp_devel_stats, default_path, datadir)
            if score>round_best_score:
               round_best_score=score
               round_best_config=candidate
               round_best_key=key
      if round_best_config is not None:
         improved=True
         current_best = round_best_config
         best_score=round_best_score
         remaining.remove(round_best_key)
         if round_best_key == "pretrained_emb_path" and "embLWsize" in remaining:
            remaining.remove("embLWsize")
         if best_model_path and os.path.exists(best_model_path):
            shutil.rmtree(best_model_path)
         if best_devel_out and os.path.exists(best_devel_out):
            os.remove(best_devel_out)
         if best_devel_stats and os.path.exists(best_devel_stats):
            os.remove(best_devel_stats)
         best_name ="best_" + build_name(current_best).replace("/", "-").replace("\\", "-").replace("pretrained_emb_path", "pre")
         best_name = f"{best_name}_{SEED}"
         best_model_path = os.path.join(models_dir, best_name)
         best_devel_out = os.path.join(results_dir, f"devel-{best_name}.out")
         best_devel_stats = os.path.join(results_dir, f"devel-{best_name}.stats")
         if os.path.exists(temp_model_path):
               shutil.rmtree(temp_model_path)
         if os.path.exists(temp_devel_out):
            os.remove(temp_devel_out)
         if os.path.exists(temp_devel_stats):
            os.remove(temp_devel_stats)
         run_single_experiment(current_best, temp_model_path, temp_devel_out, temp_devel_stats, default_path, datadir)
         copy_temporal_paths(temp_model_path, temp_devel_out, temp_devel_stats, best_model_path, best_devel_out, best_devel_stats)
      else:
         print("No improvement found in this round. Stopping.")
   best_test_out=os.path.join(results_dir, f"test-{best_name}.out")
   best_test_stats=os.path.join(results_dir, f"test-{best_name}.stats")
   predict(best_model_path, os.path.join(default_path, "preprocessed", "test.pck"), best_test_out, batch_size=current_best["batch_size"], max_len=current_best["max_len"], suf_len=current_best["suf_len"],pref_len=current_best["pref_len"])
   evaluate("NER", os.path.join(datadir, "test.xml"), best_test_out, best_test_stats)
   print("Greedy search finished")
   print(f"Best score on devel: {best_score}")
   print(f"Best model saved as: {best_model_path}")
   print(f"Best devel stats: {best_devel_stats}")
   print(f"Best test stats: {best_test_stats}")
   return current_best, best_score


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py train.pck devel.pck modelname [batch_size=N] [max_len=N] [suf_len=N]
## --

if __name__ == "__main__" :
   train_with_different_hyperparameters()
   # greedy_best_hyperparameters()
   #print("Use run.py to launch training.")
