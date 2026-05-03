from CRF import *
from MEM import *
from LinearSVM import *
from SGD import *
from PassiveAggressive import *
from dataset import Dataset
from predict import predict_selected
import paths
import os
import sys

sys.path.append(paths.UTIL)
from evaluator import evaluate

#Make sure the folders exists
os.makedirs(paths.MODELS, exist_ok=True)
os.makedirs(paths.RESULTS, exist_ok=True)

"""
CRF
"""
params= {}
crf_full= CRF(os.path.join(paths.MODELS, "model.CRF"))
info =crf_full.tagger.info()
unique_features= set()
for (feat, label) in info.state_features:
    unique_features.add(feat)
print("Unique features:", len(unique_features))

#Extract the top x features
top_k= 3500
percentile=60
top= crf_full.top_features(top_k=top_k, percentile = percentile)
selected= set(feat for feat, score in top)
print("Selected features:", len(selected))
#Train with the selected features
if percentile is not None:
    model_path = os.path.join(paths.MODELS, f"model.CRF_selected_p_{percentile}")
else:
    model_path = os.path.join(paths.MODELS, f"model.CRF_selected_{top_k}")
crf_small = CRF(model_path, params=params)
crf_small.train_with_selected_features(os.path.join(paths.PREPROCESS, "train.feat"), selected)

#Devel prediction
if percentile is not None:
    outfile= os.path.join(paths.RESULTS, f"devel-CRF_selected_p_{percentile}.out")
else:
    outfile= os.path.join(paths.RESULTS, f"devel-CRF_selected_{top_k}.out")
if percentile is not None:
    statsfile= os.path.join(paths.RESULTS, f"devel-CRF_selected_p_{percentile}.stats")
else:
    statsfile= os.path.join(paths.RESULTS, f"devel-CRF_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "devel.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "devel.xml"),outfile,statsfile)

#Test prediction
if percentile is not None:
    outfile= os.path.join(paths.RESULTS, f"test-CRF_selected_p_{percentile}.out")
else:
    outfile= os.path.join(paths.RESULTS, f"test-CRF_selected_{top_k}.out")
if percentile is not None:
    statsfile= os.path.join(paths.RESULTS, f"test-CRF_selected_p_{percentile}.stats")
else:
    statsfile= os.path.join(paths.RESULTS, f"test-CRF_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "test.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "test.xml"),outfile,statsfile)


"""
MEM
"""
params={}
mem_full=MEM(os.path.join(paths.MODELS, "model.MEM"))

#Extract the top x features
top_k=10000
percentile=85
top=mem_full.top_features(top_k=top_k, percentile=percentile)
selected =set(feat for feat, score in top)
print("Selected features:", len(selected))
#Train with the selected features
if percentile is not None:
    model_path =os.path.join(paths.MODELS, f"model.MEM_selected_p2_{percentile}")
else:
    model_path= os.path.join(paths.MODELS, f"model.MEM_selected_{top_k}")
mem_small=MEM(model_path, params=params)
mem_small.train_with_selected_features(os.path.join(paths.PREPROCESS, "train.feat"), selected)


#Devel prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"devel-MEM_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"devel-MEM_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"devel-MEM_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"devel-MEM_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "devel.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "devel.xml"),outfile,statsfile)

#Test prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"test-MEM_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"test-MEM_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"test-MEM_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"test-MEM_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "test.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "test.xml"),outfile,statsfile)

"""
LinearSVM
"""
params={}
linearsvm_full=LinearSVM(os.path.join(paths.MODELS, "model.LinearSVM"))

#Extract the top x features
# top_k=5000
# percentile=None
top=linearsvm_full.top_features(top_k=top_k, percentile = percentile)
selected=set(feat for feat, score in top)
print("Selected features:", len(selected))
#Train with the selected features
if percentile is not None:
    model_path=os.path.join(paths.MODELS, f"model.LinearSVM_selected_p2_{percentile}")
else:
    model_path=os.path.join(paths.MODELS, f"model.LinearSVM_selected_{top_k}")
linearsvm_small=LinearSVM(model_path, params=params)
linearsvm_small.train_with_selected_features(os.path.join(paths.PREPROCESS, "train.feat"), selected)


#Devel prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"devel-LinearSVM_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"devel-LinearSVM_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"devel-LinearSVM_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"devel-LinearSVM_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "devel.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "devel.xml"),outfile,statsfile)

#Test prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"test-LinearSVM_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"test-LinearSVM_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"test-LinearSVM_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"test-LinearSVM_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "test.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "test.xml"),outfile,statsfile)


"""
SGD
"""
params={}
sgd_full=SGD(os.path.join(paths.MODELS, "model.SGD"))


#Extract the top x features
# top_k=5000
# percentile=None
top=sgd_full.top_features(top_k=top_k, percentile = percentile)
selected=set(feat for feat, score in top)
print("Selected features:", len(selected))
#Train with the selected features
if percentile is not None:
    model_path=os.path.join(paths.MODELS, f"model.SGD_selected_p2_{percentile}")
else:
    model_path=os.path.join(paths.MODELS, f"model.SGD_selected_{top_k}")
sgd_small=SGD(model_path, params=params)
sgd_small.train_with_selected_features(os.path.join(paths.PREPROCESS, "train.feat"), selected)


#Devel prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"devel-SGD_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"devel-SGD_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"devel-SGD_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"devel-SGD_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "devel.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "devel.xml"),outfile,statsfile)

#Test prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"test-SGD_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"test-SGD_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"test-SGD_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"test-SGD_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "test.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "test.xml"),outfile,statsfile)

"""
PassiveAggressive
"""
params={}
passiveaggressive_full=PassiveAggressive(os.path.join(paths.MODELS, "model.PassiveAggressive"))


#Extract the top x features
# top_k = 5000
# percentile = None
top=passiveaggressive_full.top_features(top_k=top_k, percentile = percentile)
selected=set(feat for feat, score in top)
print("Selected features:", len(selected))
#Train with the selected features
if percentile is not None:
    model_path=os.path.join(paths.MODELS, f"model.PassiveAggressive_selected_p2_{percentile}")
else:
    model_path=os.path.join(paths.MODELS, f"model.PassiveAggressive_selected_{top_k}")
passiveaggressive_small=PassiveAggressive(model_path, params=params)
passiveaggressive_small.train_with_selected_features(os.path.join(paths.PREPROCESS, "train.feat"), selected)


#Devel prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"devel-PassiveAggressive_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"devel-PassiveAggressive_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"devel-PassiveAggressive_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"devel-PassiveAggressive_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "devel.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "devel.xml"),outfile,statsfile)

#Test prediction
if percentile is not None:
    outfile=os.path.join(paths.RESULTS, f"test-PassiveAggressive_selected_p2_{percentile}.out")
else:
    outfile=os.path.join(paths.RESULTS, f"test-PassiveAggressive_selected_{top_k}.out")
if percentile is not None:
    statsfile=os.path.join(paths.RESULTS, f"test-PassiveAggressive_selected_p2_{percentile}.stats")
else:
    statsfile=os.path.join(paths.RESULTS, f"test-PassiveAggressive_selected_{top_k}.stats")
predict_selected(os.path.join(paths.PREPROCESS, "test.feat"),model_path,outfile,selected)
evaluate("NER",os.path.join(paths.DATA, "test.xml"),outfile,statsfile)
