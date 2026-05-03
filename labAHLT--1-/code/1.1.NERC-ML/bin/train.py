#!/usr/bin/env python3

import sys
from MEM import *
from SVM import *
from CRF import *
from PassiveAggressive import *
from SGD import *
from LinearSVM import *

def train(datafile, params, modelfile) :
    # Create an empty model of the appropriate type
    model_type = modelfile.split(".")[-1].lower()
    if model_type == "mem" : model = MEM(modelfile, params)
    elif model_type == "svm" : model = SVM(modelfile, params)
    elif model_type == "crf" : model = CRF(modelfile, params) 
    elif model_type == "sgd": model = SGD(modelfile, params)
    elif model_type == "passiveaggressive": model = PassiveAggressive(modelfile, params)
    elif model_type == "linearsvm": model = LinearSVM(modelfile, params)      
    else :
        print(f"Invalid model type '{model_type}'")
        sys.exit(1)

    # Train and store the model
    model.train(datafile)


if __name__ == "__main__" :
    # get file where model will be written
    datafile = sys.argv[1]
    modelfile = sys.argv[2]
    
    # get parameters in line.  e.g. C=10 kernel=rbf degree=2
    params = {}
    pars = sys.argv[3:]
    for x in pars:
        par,val = x.split("=")
        params[par] = val

    # train model and store in given filename
    train(datafile, params, modelfile)
