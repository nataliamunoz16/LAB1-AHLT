#####################################################
## Class to store an LinearSVC model
#####################################################
import pickle
import sys
import scipy
import numpy as np
from sklearn.svm import LinearSVC

import dataset

class LinearSVM:

    ## --------------------------------------------------
    ## Constructor: Load model from file
    ## --------------------------------------------------
    def __init__(self, modelfile, params=None):

        self.modelfile = modelfile
        
        if params is None :
            # only modelfile given, assume it is an existing model and load it        
            with open(self.modelfile, 'rb') as df :
                self.tagger = pickle.load(df)
            with open(self.modelfile+".idx", 'rb') as df :
                self.fidx = pickle.load(df)

        else : # params given, create new empty model

            # extract parameters if provided. Use default if not
            C = float(params['C']) if 'C' in params else 1.0
            max_iter = int(params['max_iter']) if 'max_iter' in params else 5000
            tol = float(params['tol']) if 'tol' in params else 1e-4
            if 'dual' in params:
                dual_value = params['dual']
                if isinstance(dual_value, str):
                    dual = dual_value.lower() == 'true'
                else:
                    dual = bool(dual_value)
            else:
                dual = True
            # create classifier
            self.tagger = LinearSVC(
                            C=C,
                            max_iter=max_iter,
                            tol=tol,
                            dual=dual,
                            verbose=1
                            )


    ## --------------------------------------------------
    ## train a model on given data, store in modelfile
    ## --------------------------------------------------
    def train(self, datafile):
        # load dataset
        ds = dataset.Dataset(datafile)
        self.fidx = ds.feature_index()

        # Read training instances 
        X,Y = ds.csr_matrix()

        # train classifier
        self.tagger.fit(X,Y)

        # save model
        pickle.dump(self.tagger, open(self.modelfile, 'wb'))
        pickle.dump(self.fidx, open(self.modelfile+".idx", 'wb'))
    

    ## --------------------------------------------------
    ## predict best class for each element in xseq
    ## --------------------------------------------------
    def predict(self, xseq):
        if len(xseq)==0 : return []
        
        # Encode xseq into a CSR sparse matrix
        rowi = [] # row (example number)
        colj = [] # column (feature number)
        data = [] # value (1 or 0 since we use binary features)
        nex = 0 # example  counter (each word is one example)
        for w in xseq :
            for f in w :
                if f in self.fidx :
                    data.append(1)
                    rowi.append(nex)
                    colj.append(self.fidx[f]) 
                    # next word           
            nex += 1
        X = scipy.sparse.csr_matrix((data, (rowi, colj)), shape=(len(xseq),len(self.fidx)))
        
        # apply model to X and return predictions
        return self.tagger.predict(X)
    
    def top_features(self, top_k=100, percentile=None):
        """
        Returns the most important features according to the absolute weights of the model
        """
        if self.tagger is None:
            print("This model has not been trained", file=sys.stderr)
            sys.exit(1)
        if not hasattr(self, "fidx") or self.fidx is None:
            print("Feature index not found", file=sys.stderr)
            sys.exit(1)
        if not hasattr(self.tagger, "coef_"):
            print("This model does not provide feature coefficients", file=sys.stderr)
            sys.exit(1)
        #Invert the indeces of the features
        inv_fidx={idx: feat for feat, idx in self.fidx.items()}
        #Obtain the weights of the model
        weights=np.abs(self.tagger.coef_)
        #Obtain the max for the different classes
        if weights.ndim== 2:
            feat_scores =weights.max(axis=0)
        else:
            feat_scores= np.abs(weights)
        # Construct a list of (feature, score)
        feat_score=[(inv_fidx[i], feat_scores[i]) for i in range(len(feat_scores)) if i in inv_fidx]
        feat_score.sort(key=lambda x: x[1], reverse=True)
        if percentile is not None:
            scores = [score for _, score in feat_score if score > 0]
            threshold= np.percentile(scores, percentile)
            selected =[(feat, score) for feat, score in feat_score if score >= threshold]
            return selected
        return feat_score[:top_k]


    def filter_xseq(self, xseq, selected_features):
        """
        Removes from each token the features that are not in the selected ones
        """
        return [[feat for feat in token_features if feat in selected_features] for token_features in xseq]


    def train_with_selected_features(self, datafile, selected_features):
        """
        Retrains the model using only the selected features
        """
        #Load the dataset
        ds=dataset.Dataset(datafile)
        #Obtain all the features
        all_fidx=ds.feature_index()
        #Extract the selected ones
        selected_feat=[feat for feat in all_fidx if feat in selected_features]
        #Create new index
        self.fidx={feat: idx for idx, feat in enumerate(selected_feat)}
        #Initialize the lists to create the matrix
        rowi=[]
        colj=[]
        data=[]
        Y=[]
        number_ex=0
        #For each example
        for xseq, yseq, _ in ds.instances():
            #Filter the features
            filtered_xseq=self.filter_xseq(xseq, selected_features)
            #For each token add the feature to the matrix
            for feats, label in zip(filtered_xseq, yseq):
                for f in feats:
                    if f in self.fidx:
                        rowi.append(number_ex)
                        colj.append(self.fidx[f])
                        data.append(1)
                Y.append(label)
                number_ex += 1
        #Create the sparse matrix
        X= scipy.sparse.csr_matrix((data, (rowi, colj)),shape=(len(Y),len(self.fidx)))
        #Train the model
        self.tagger.fit(X, Y)
        #Save the model and the indices of the features
        pickle.dump(self.tagger, open(self.modelfile, 'wb'))
        pickle.dump(self.fidx, open(self.modelfile + ".idx", 'wb'))
