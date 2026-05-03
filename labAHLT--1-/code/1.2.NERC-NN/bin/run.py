#! /usr/bin/python3

import sys
import os
import argparse
from dataset import Dataset
from train import do_train
from predict import predict

##########################################################
#
#  This script allows to run a series of experiments
#  on NER on medical text using NN
#
#  You can select wich steps of the experiment execcute:
#    - parse: Use spaccy to parse the documents and store results in pickle files
#    - train: Train a NN model
#    - predict: Apply the model to development data set and evaluate performance
#
#  You can add hyperparameters for training
#    - batch_size, max_len, suf_len, epochs, optimizer, lr, embLWsize, embWsize, embSsize, dropout_rate, lstm_out_size, linear_out_size, num_layers_lstm, dropout_lstm, linear2
#    Omitted parameters will receive a default value
#    Parametres may be mixed, each model will select its own.
#
#  Examples:
#
#      # Extract features, train, and evaluate a CRF model
#      python3 run.py parse --name mymodel_001 --batch_size 32 --max_len 140     
#      python3 run.py train --name mymodel_001 --batch_size 32 --max_len 140 --epochs 5
#      python3 run.py predict --name mymodel_001
#      
#      # the 3 lines above can be run in a single one:
#      python3 run.py parse train predict name=mymodel_001 batch_size=32 max_len=140
##

BINDIR=os.path.abspath(os.path.dirname(__file__)) # location of this file
NERDIR=os.path.dirname(BINDIR) # one level up
SOLDIR=os.path.dirname(NERDIR) # one level up
MAINDIR=os.path.dirname(SOLDIR) # one level up
DATADIR=os.path.join(MAINDIR,"data") # down to "data"
UTILDIR=os.path.join(MAINDIR,"util") # down to "util"

sys.path.append(UTILDIR)
from evaluator import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Run NER experiments")
    parser.add_argument("steps",nargs="+",choices=["parse", "train", "predict", "test"],help="Steps to execute")
    parser.add_argument("--name", default="mymodel_000", help="Model name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=150)
    parser.add_argument("--suf_len", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "AdamW"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embLWsize", type=int, default=100)
    parser.add_argument("--embWsize", type=int, default=100)
    parser.add_argument("--embSsize", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--lstm_out_size", type=int, default=200)
    parser.add_argument("--linear_out_size", type=int, default=200)
    parser.add_argument("--num_layers_lstm", type=int, default=1)
    parser.add_argument("--dropout_lstm", action="store_true")
    parser.add_argument("--linear2", action="store_true")
    
    parser.add_argument("--pretrained_emb_path", default=None)
    parser.add_argument("--freeze_pretrained", action="store_true")
    parser.add_argument("--activation", default='relu', choices=['relu','tanh','gelu','leaky','elu'])
    return parser.parse_args()

args = parse_args()

# if feature extraction is required, do it
if "parse" in args.steps:
    os.makedirs(os.path.join(NERDIR, "preprocessed"), exist_ok=True)
    # if test is required, extract features from test
    if "test" in args.steps:
        print("Creating parsed test pickle file...         ")
        ds = Dataset(os.path.join(DATADIR,"test.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed","test.pck"))

    else: # otherwise, extract features for train and devel
        # convert datasets to feature vectors
        print("Creating parsed train pickle file...         ")
        ds = Dataset(os.path.join(DATADIR,"train.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed","train.pck"))
        print("Creating parsed devel pickle file...         ")
        ds = Dataset(os.path.join(DATADIR,"devel.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed","devel.pck"))

# for each required model, see if training or prediction are required

if "train" in args.steps:
    os.makedirs(os.path.join(NERDIR,"models"), exist_ok=True)
    # train model
    print(f"Training model {args.name} ...")
    do_train(
        os.path.join(NERDIR, "preprocessed", "train.pck"),
        os.path.join(NERDIR, "preprocessed", "devel.pck"),
        modelname=os.path.join(NERDIR, "models", args.name),
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.lr,
        embLWsize=args.embLWsize,
        embWsize=args.embWsize,
        embSsize=args.embSsize,
        dropout_rate=args.dropout_rate,
        lstm_out_size=args.lstm_out_size,
        linear_out_size=args.linear_out_size,
        num_layers_lstm=args.num_layers_lstm,
        max_len=args.max_len,
        suf_len=args.suf_len,
        dropout_lstm=args.dropout_lstm,
        linear2=args.linear2,
        pretrained_emb_path= args.pretrained_emb_path,
        freeze_pretrained=args.freeze_pretrained,
        activation=args.activation
    )

if "predict" in args.steps:    
    os.makedirs(os.path.join(NERDIR,"results"), exist_ok=True)
    if "test" in args.steps:
        # run model on test data and evaluate results
        print(f"Running {args.name} model on test...")
        out_file = os.path.join(NERDIR, "results", f"test-{args.name}.out")

        predict(
            os.path.join(NERDIR, "models", args.name),
            os.path.join(NERDIR, "preprocessed", "test.pck"),
            out_file,
            batch_size=args.batch_size,
            max_len=args.max_len,
            suf_len=args.suf_len
        )
        evaluate(
            "NER",
            os.path.join(DATADIR, "test.xml"),
            out_file,
            os.path.join(NERDIR, "results", f"test-{args.name}.stats")
        )
    else:
        # run model on devel data and evaluate results
        out_file = os.path.join(NERDIR, "results", f"devel-{args.name}.out")
        predict(
            os.path.join(NERDIR, "models", args.name),
            os.path.join(NERDIR, "preprocessed", "devel.pck"),
            out_file,
            batch_size=args.batch_size,
            max_len=args.max_len,
            suf_len=args.suf_len
        )
        evaluate(
            "NER",
            os.path.join(DATADIR, "devel.xml"),
            out_file,
            os.path.join(NERDIR, "results", f"devel-{args.name}.stats")
        )
# C:\Users\Rafael\Downloads\glove.2024.wikigiga.100d\wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt
'''
        # run model on train data and evaluate results
        print(f"Running {params['name']} model on train...")
        predict(os.path.join(NERDIR, "models",params["name"]),
                os.path.join(NERDIR, "preprocessed", "train.pck"),
                params,
                os.path.join(NERDIR,"results","train-"+model+".out"))
        evaluate("NER", os.path.join(DATADIR,"train.xml"),
                 os.path.join(NERDIR,"results","train-"+model+".out"),
                 os.path.join(NERDIR,"results","train-"+model+".stats"))
        '''
'''
for act in ['relu', 'tanh','gelu', 'leaky', 'elu']:
    # Dynamically change the name so results don't overwrite each other
    current_name = f"{args.name}_{act}"
    print(f"\n{'='*20}\nRunning experiment with: {act}\n{'='*20}")

    if "train" in args.steps:
        os.makedirs(os.path.join(NERDIR, "models"), exist_ok=True)
        print(f"Training model {current_name} ...")
        do_train(
            os.path.join(NERDIR, "preprocessed", "train.pck"),
            os.path.join(NERDIR, "preprocessed", "devel.pck"),
            modelname=os.path.join(NERDIR, "models", current_name),
            activation=act,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            lr=args.lr,
            embLWsize=args.embLWsize,
            embWsize=args.embWsize,
            embSsize=args.embSsize,
            dropout_rate=args.dropout_rate,
            lstm_out_size=args.lstm_out_size,
            linear_out_size=args.linear_out_size,
            num_layers_lstm=args.num_layers_lstm,
            max_len=args.max_len,
            suf_len=args.suf_len,
            dropout_lstm=args.dropout_lstm,
            linear2=args.linear2,
            pretrained_emb_path=args.pretrained_emb_path,
            freeze_pretrained=args.freeze_pretrained
        )

    if "predict" in args.steps:
        os.makedirs(os.path.join(NERDIR, "results"), exist_ok=True)
        
        # Use current_name for out_file and model path
        prefix = "test" if "test" in args.steps else "devel"
        data_pck = "test.pck" if "test" in args.steps else "devel.pck"
        xml_file = "test.xml" if "test" in args.steps else "devel.xml"
        
        out_file = os.path.join(NERDIR, "results", f"{prefix}-{current_name}.out")
        stats_file = os.path.join(NERDIR, "results", f"{prefix}-{current_name}.stats")

        predict(
            os.path.join(NERDIR, "models", current_name),
            os.path.join(NERDIR, "preprocessed", data_pck),
            out_file,
            batch_size=args.batch_size,
            max_len=args.max_len,
            suf_len=args.suf_len
        )
        evaluate(
            "NER",
            os.path.join(DATADIR, xml_file),
            out_file,
            stats_file
        )'''