#####################################################
## Class to store an ngram ME model
#####################################################
import sys
import pickle

import scipy
import sklearn
from sklearn.svm import SVC

import dataset

class SVM:

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
            kernel = params['kernel'] if 'kernel' in params else 'rbf'
            degree = int(params['degree']) if 'degree' in params else 3
            gamma = params['gamma'] if 'gamma' in params else 'scale'
                
            # create classifier
            self.tagger = SVC(verbose=True,
                            C=C,
                            kernel=kernel,
                            degree=degree,
                            gamma=gamma
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
