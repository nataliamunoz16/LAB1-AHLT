#####################################################
## Class to store an ngram ME model
#####################################################

import sys
import pycrfsuite
import numpy as np
from dataset import *
from collections import defaultdict

class CRF:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile=None, params=None):

        self.modelfile = modelfile
        self.tagger = None
        self.trainer = None
        if params is None:
            # only modelfile given, assume it is an existing model and load it        
            # modelfile given, assume it is an existing model and load it
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(self.modelfile)
                
        else :  # params given, create new empty model

            # extract parameters if provided. Use default if not
            alg = params['algorithm'] if 'algorithm' in params else 'lbfgs'
            minf = int(params['feature.minfreq']) if 'feature.minfreq' in params else 1
            maxit =  int(params['max_iterations']) if 'max_iterations' in params else 9999999
            c1 = float(params['c1']) if 'c1' in params else 0.1
            c2 = float(params['c2']) if 'c2' in params else 1.0
            eps = float(params['epsilon']) if 'epsilon' in params else 0.00001
            # select needed parametes depending on the agorithm
            params = {'feature.minfreq' : minf, 'max_iterations' : maxit}
            if alg == "lbfgs" : params['c1'] = c1
            if alg in ["lbfgs", "l2sgd"] : params['c2'] = c2
            if alg != "l2sgd" : params['epsilon'] = eps
            # create and train empty classifier with given algorithm and parameters
            self.trainer = pycrfsuite.Trainer(alg, params)

    ## --------------------------------------------------
    ## train a model on given data, store in modelfile
    ## --------------------------------------------------
    def train(self, datafile):
        # load dataset
        ds = Dataset(datafile)
        # add examples to trainer
        for xseq, yseq, _ in ds.instances() :
            self.trainer.append(xseq, yseq, 0)

        # train and store model 
        self.trainer.train(self.modelfile, -1)

        
    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, xseq):
        if self.tagger is None :
            print("This model has not been trained", file=sys.stderr)
            sys.exit(1)

        return self.tagger.tag(xseq)
    
    def top_features(self, top_k=100, percentile=None):
        """
        Returns the most important features according to the absolute weights of the model
        """
        if self.tagger is None:
            print("This model has not been trained", file=sys.stderr)
            sys.exit(1)
        #Access the model information to extract the feature weights
        info =self.tagger.info()
        #Store the maximum absolite for each feature
        feat_scores= defaultdict(float)
        for (feat, label), weight in info.state_features.items():
            feat_scores[feat]=max(feat_scores[feat], abs(weight))
        #Sort by importance
        ranked=sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)
        #Select based on percentile threshold
        if percentile is not None:
            scores= [score for _, score in ranked if score > 0]
            threshold =np.percentile(scores, percentile)
            selected= [(feature, score) for feature, score in ranked if score >= threshold]
            return selected
        return ranked[:top_k]
    
    def filter_xseq(self, xseq, selected_features):
        """
        Removes from each token the features that are not in the selected ones
        """
        if selected_features is None:
            return [[feat for feat in token_features] for token_features in xseq]
        return [[feat for feat in token_features if feat in selected_features] for token_features in xseq]
    
    def train_with_selected_features(self, datafile, selected_features):
        """
        Retrains the model using only the selected features
        """
        #Load dataset
        ds=Dataset(datafile)
        #Keep only the selected features for each token
        if selected_features is not None:
            print('selected', selected_features)
            for xseq, yseq, _ in ds.instances():
                filtered_xseq=self.filter_xseq(xseq, selected_features)
                self.trainer.append(filtered_xseq, yseq, 0)
        else:
            for xseq, yseq, _ in ds.instances():
                self.trainer.append(xseq, yseq, 0)
        #Train with the selected features
        self.trainer.train(self.modelfile, -1)
        self.tagger=pycrfsuite.Tagger()
        self.tagger.open(self.modelfile)
    

