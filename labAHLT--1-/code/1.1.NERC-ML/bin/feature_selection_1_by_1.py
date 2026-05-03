import os
import sys
import json
import time
#Import the models to test
from CRF import CRF
from MEM import MEM
from SVM import SVM
from LinearSVM import LinearSVM
from SGD import SGD
from PassiveAggressive import PassiveAggressive

from predict import predict_selected
import paths
#Add evaluator folder
sys.path.append(paths.UTIL)
from evaluator import evaluate

#Number of metadata columns before the features
META_COLUMNS=5

#Create output folders
os.makedirs(paths.MODELS, exist_ok=True)
os.makedirs(paths.RESULTS, exist_ok=True)
os.makedirs(paths.PREPROCESS, exist_ok=True)

def get_feature_groups_by_prefix(features, prefixes):
    """
    Create a dictionary with the prefix and the set of features starting with that prefix
    """
    groups={}
    for prefix in prefixes:
        groups[prefix]=set()
    for feat in features:
        for prefix in prefixes:
            if feat.startswith(prefix):
                groups[prefix].add(feat)
    return groups

def read_features_file(filename):
    """
    Read a .feat file and store it in a list
    """
    data=[]
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            #Empty line is a sentence separator
            if line == "":
                data.append(None)
            else:
                parts=line.split("\t")
                meta=parts[:META_COLUMNS]
                feats=parts[META_COLUMNS:]
                data.append((meta, feats))
    return data

def get_all_distinct_features_from_data(data):
    """
    Extract the distinct features appearing in a .feat file
    """
    distinct_features=set()
    for row in data:
        if row is None:
            continue
        meta, feats=row
        for feat in feats:
            distinct_features.add(feat)
    return distinct_features

def check_same_tokens(data1, data2):
    """
    Check that two .feat files have the same tokens
    """
    if len(data1) != len(data2):
        raise ValueError("the files have different number of tokens")
    for i in range(len(data1)):
        row1=data1[i]
        row2=data2[i]
        #If both lines are sentence separators
        if row1 is None and row2 is None:
            continue
        if row1 is None or row2 is None:
            raise ValueError(f"Error in the line {i+1}")
        meta1=row1[0]
        meta2=row2[0]
        if meta1 != meta2:
            raise ValueError(f"Different tokens in line {i+1}: {meta1} != {meta2}")

def get_features_with_prefixes(features, prefixes):
    """
    From a set of features keep those that start with a preffix
    """
    selected=set()
    for feature in features:
        if feature.startswith(tuple(prefixes)):
            selected.add(feature)
    return selected

def write_feat_file(data, allowed_features, outfile):
    """
    Write a new .feat file keeping the selected features
    """
    with open(outfile, "w", encoding="utf-8") as out:
        for row in data:
            if row is None:
                out.write("\n")
            else:
                meta, features=row
                kept=[f for f in features if f in allowed_features]
                out.write("\t".join(meta + kept) + "\n")

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

def create_model(model_name, model_path):
    """
    Create and return the model object
    """
    params = {} #initialize the params epmpty
    if model_name == "CRF":
        return CRF(model_path, params=params)
    elif model_name == "MEM":
        return MEM(model_path, params=params)
    elif model_name == "SVM":
        return SVM(model_path, params=params)
    elif model_name == "LinearSVM":
        return LinearSVM(model_path, params=params)
    elif model_name == "PassiveAggressive":
        return PassiveAggressive(model_path, params=params)
    elif model_name == "SGD":
        return SGD(model_path, params=params)
    else:
        raise ValueError("Model no valid. Use CRF, MEM, SGD, PassiveAggressive or LinearSVM")

def train_and_evaluate(model_name, train_file, devel_file):
    """
    Train a model on 'train_file' and evaluates it on 'devel_file'
    """
    model_path=os.path.join(paths.MODELS, f"model.temp.{model_name}")
    outfile=os.path.join(paths.RESULTS, f"temp_{model_name}.out")
    statsfile=os.path.join(paths.RESULTS, f"temp_{model_name}.stats")
    #Create model
    model=create_model(model_name, model_path)
    #Extract features from the training
    train_data=read_features_file(train_file)
    selected_features=get_all_distinct_features_from_data(train_data)
    #Train the model using the selected features
    model.train_with_selected_features(train_file, selected_features)
    # Predict labels for the development file and evaluate them
    predict_selected(devel_file, model_path, outfile, selected_features)
    evaluate("NER", os.path.join(paths.DATA, "devel.xml"), outfile, statsfile)
    #Read the final F1
    f1=read_f1_from_stats(statsfile)
    return f1

def select_best_features_for_model_greedy(model_name,candidate_groups,orig_train_data,full_train_data,full_devel_data,full_test_data):
    """
    Performs greedy feature selection
    """
    print(f"\n MODEL: {model_name} ")
    #Starts from the original features and creates a baseline
    accepted_features=get_all_distinct_features_from_data(orig_train_data)
    baseline_train=os.path.join(paths.PREPROCESS, f"baseline_train_{model_name}.feat")
    baseline_devel=os.path.join(paths.PREPROCESS, f"baseline_devel_{model_name}.feat")
    write_feat_file(full_train_data,accepted_features,baseline_train)
    write_feat_file(full_devel_data,accepted_features,baseline_devel)
    #Train and evaluate the baseline
    best_score=train_and_evaluate(model_name,baseline_train, baseline_devel)
    print("Baseline F1:", best_score)
    accepted_prefixes=[]
    remaining_prefixes=list(candidate_groups.keys())
    max_prefixes=10
    iteration=0
    #Try candidates one by one
    while remaining_prefixes and iteration < max_prefixes:
        best_prefix=None
        best_iteration_score=best_score
        for prefix in remaining_prefixes:
            group_features=candidate_groups[prefix]
            temp_features=set(accepted_features)
            temp_features.update(group_features)
            temp_train=os.path.join(paths.PREPROCESS,f"temp_train_{model_name}.feat")
            temp_devel=os.path.join(paths.PREPROCESS,f"temp_devel_{model_name}.feat")
            write_feat_file(full_train_data,temp_features,temp_train)
            write_feat_file(full_devel_data,temp_features,temp_devel)
            start=time.time()
            score=train_and_evaluate(model_name,temp_train,temp_devel)
            print(f"{prefix} -> F1 {score} (time {time.time()-start:.2f}s)")
            if score>best_iteration_score:
                best_iteration_score=score
                best_prefix=prefix
        if best_prefix is None:
            print("No improvement.Stopping.")
            break
        print(f"\nSelected prefix: {best_prefix}")
        print("New F1:",best_iteration_score)
        accepted_prefixes.append(best_prefix)
        accepted_features.update(candidate_groups[best_prefix])
        remaining_prefixes.remove(best_prefix)
        best_score=best_iteration_score
        iteration+=1
    #Final selected files
    selected_train=os.path.join(paths.PREPROCESS, f"selected_train_{model_name}.feat")
    selected_devel = os.path.join(paths.PREPROCESS, f'selected_devel_{model_name}.feat')
    selected_test = os.path.join(paths.PREPROCESS, f"selected_test_{model_name}.feat")
    write_feat_file(full_train_data, accepted_features, selected_train)
    write_feat_file(full_devel_data, accepted_features, selected_devel)
    write_feat_file(full_test_data, accepted_features, selected_test)

    print(f"Model {model_name} finished")
    print("Selected prefixes:",accepted_prefixes)
    print("Final F1:",best_score)
    return accepted_prefixes,accepted_features,best_score

def train_final_model_and_evaluate(model_name, selected_train_file, selected_devel_file, selected_test_file, selected_features):
    """
    Train the final model with the selected features, save it, and evaluate on devel and test
    """
    model_path = os.path.join(paths.MODELS, f"best_model.{model_name}")
    devel_outfile = os.path.join(paths.RESULTS, f"best_{model_name}_devel.out")
    devel_statsfile = os.path.join(paths.RESULTS, f"best_{model_name}_devel.stats")
    test_outfile = os.path.join(paths.RESULTS, f"best_{model_name}_test.out")
    test_statsfile = os.path.join(paths.RESULTS, f"best_{model_name}_test.stats")
    #Create model
    model = create_model(model_name, model_path)
    #train final model
    model.train_with_selected_features(selected_train_file, selected_features)
    #Evaluate on devel
    predict_selected(selected_devel_file, model_path, devel_outfile, selected_features)
    evaluate("NER", os.path.join(paths.DATA, "devel.xml"), devel_outfile, devel_statsfile)
    devel_f1 = read_f1_from_stats(devel_statsfile)
    #Evaluate on test
    predict_selected(selected_test_file, model_path, test_outfile, selected_features)
    evaluate("NER", os.path.join(paths.DATA, "test.xml"), test_outfile, test_statsfile)
    test_f1 = read_f1_from_stats(test_statsfile)

    return {"model_path": model_path,"devel_output": devel_outfile,"devel_stats": devel_statsfile,"devel_f1": devel_f1,"test_output": test_outfile,"test_stats": test_statsfile,"test_f1": test_f1}

def select_best_features_for_model(model_name,candidate_groups,orig_train_data,orig_devel_data,orig_test_data,full_train_data,full_devel_data,full_test_data):
    """
    Performs greedy feature selection
    """
    print(f"\n MODEL: {model_name} ")
    #Starts from the original features and creates a baseline
    accepted_features=get_all_distinct_features_from_data(orig_train_data)
    baseline_train=os.path.join(paths.PREPROCESS, f"baseline_train_{model_name}.feat")
    baseline_devel=os.path.join(paths.PREPROCESS, f"baseline_devel_{model_name}.feat")
    write_feat_file(full_train_data,accepted_features,baseline_train)
    write_feat_file(full_devel_data,accepted_features,baseline_devel)
    #Train and evaluate the baseline
    best_score=train_and_evaluate(model_name,baseline_train, baseline_devel)
    print("Baseline F1:", best_score)
    #Store the accepted features
    accepted_new_features=[]
    num_accepted_new=0
    #Try candidates one by one
    for prefix in candidate_groups:
        group_features= candidate_groups[prefix]
        print(f"Trying group: {prefix} ({len(group_features)} features)")
        #Temporary features and train/devel files for this trial
        temp_features =set(accepted_features)
        temp_features.update(group_features)
        temp_train=os.path.join(paths.PREPROCESS, f"temp_train_{model_name}.feat")
        temp_devel = os.path.join(paths.PREPROCESS, f"temp_devel_{model_name}.feat")
        write_feat_file(full_train_data, temp_features, temp_train)
        write_feat_file(full_devel_data, temp_features, temp_devel)
        #Train and evaluate on this extra feature
        time_init =time.time()
        score= train_and_evaluate(model_name, temp_train, temp_devel)
        print(f'Training time: {time.time() - time_init:.2f} seconds')
        #Keep it if it improves the best score
        print('F1:', score)
        if score > best_score:
            accepted_features.update(group_features)
            accepted_new_features.append(prefix)
            num_accepted_new=num_accepted_new + len(group_features)
            best_score=score
    #Final selected files
    selected_train=os.path.join(paths.PREPROCESS, f"selected_train_{model_name}.feat")
    selected_devel = os.path.join(paths.PREPROCESS, f'selected_devel_{model_name}.feat')
    selected_test = os.path.join(paths.PREPROCESS, f"selected_test_{model_name}.feat")
    write_feat_file(full_train_data, accepted_features, selected_train)
    write_feat_file(full_devel_data, accepted_features, selected_devel)
    write_feat_file(full_test_data, accepted_features, selected_test)

    print(f"Model {model_name} finished")
    print("New features preffix len:", len(accepted_new_features))
    print("New features:", num_accepted_new)
    print("F1 final:", best_score)
    print("created files:")
    print(selected_train)
    print(selected_devel)
    print(selected_test)
    return accepted_new_features, accepted_features, best_score


def run_feature_selection_for_all_models(prefixes,orig_train_file,orig_devel_file,orig_test_file,full_train_file,full_devel_file,full_test_file):
    """
    Runs the whole feature selection pipeline for CRF, MEM and SVM
    """
    #Read the original files and the full feature files
    orig_train_data=read_features_file(orig_train_file)
    orig_devel_data=read_features_file(orig_devel_file)
    orig_test_data= read_features_file(orig_test_file)
    full_train_data= read_features_file(full_train_file)
    full_devel_data= read_features_file(full_devel_file)
    full_test_data= read_features_file(full_test_file)
    #Check that the versions are aligned by token
    check_same_tokens(orig_train_data, full_train_data)
    check_same_tokens(orig_devel_data, full_devel_data)
    check_same_tokens(orig_test_data, full_test_data)
    #Extract all the features
    all_full_features=get_all_distinct_features_from_data(full_train_data)
    original_features=get_all_distinct_features_from_data(orig_train_data)
    #Extract the features with the desired prefixxes and remove the ones that are in the baseline
    candidate_features= get_features_with_prefixes(all_full_features, prefixes)
    candidate_features=candidate_features - original_features
    candidate_groups= get_feature_groups_by_prefix(candidate_features, prefixes)
    print("Candidate groups:")
    for prefix in candidate_groups:
        print(f"{prefix}: {len(candidate_groups[prefix])} features")
    print("Number of candidate groups:", len(candidate_groups))
    results = {}
    #Run the selection for each model
    for model_name in ["PassiveAggressive", "SGD", "LinearSVM"]:
        best_prefixes, selected_features, best_dev_score =select_best_features_for_model(model_name,candidate_groups,orig_train_data,orig_devel_data,orig_test_data,full_train_data,full_devel_data,full_test_data)
        # results[model_name]={"features": best_features,"score": best_score}
        selected_train = os.path.join(paths.PREPROCESS, f"selected_train_{model_name}.feat")
        selected_devel = os.path.join(paths.PREPROCESS, f"selected_devel_{model_name}.feat")
        selected_test = os.path.join(paths.PREPROCESS, f"selected_test_{model_name}.feat")
        final_result = train_final_model_and_evaluate(model_name,selected_train,selected_devel,selected_test,selected_features)
        results[model_name] = {"accepted_prefixes": best_prefixes,"selection_dev_f1": best_dev_score,"final_devel_f1": final_result["devel_f1"],"final_test_f1": final_result["test_f1"],"model_path": final_result["model_path"],"devel_output": final_result["devel_output"],"devel_stats": final_result["devel_stats"],"test_output": final_result["test_output"],"test_stats": final_result["test_stats"]}
    #Save summary results to JSON
    json_path= os.path.join(paths.RESULTS, "feature_selection_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    return results



if __name__ == "__main__":
    #Run the whole process
    results = run_feature_selection_for_all_models(
        prefixes = ["hasDot", "hasComma", "hasSlash", "hasPlus", "hasMinus", "isPunct", "isAscii", "isGreekLetter", "isCompound", "inNounChunk", "B-NP", "E-NP",  "hasInternalCap", "looksLikeFormula", "isInsideParentheses", "isParenAbbrev", "isMedicalModifier", "drugListStart", "possibleDiscontinuous", "external_in_dict", "externalpart_in_dict", "hasDash" ,"prevIsPunct", "NextIsPunct", "isDigit", "hasDigit", "isUpper", "isTitle", "isAllUpper", "isPlural", "NextIsLeftParen", "NextIsRightParen", "prevIsComma", "NextIsComma", "is_hyp_drug", "is_hyp_group", "isUnit","isAlpha", "nChildren", "prev_bigram_is_", "next_bigram_is_", "nearDict", "lenToken", "isStopword", "most_freq_class", "context", "in_missing_", "externalpart", "nearTrigger", "TopPre", "TopSuf", "tfidf_cosine_centroid", "headPos", "pos=", "childPos", "external", "headDep", "childDep", "dep=", "tag=", "cshape=", "shape=", "shapeBigramNext=", "suf3Next", "suf3Prev", "pre3Next=", "pre3Prev=", "pre3=", "suf4Next", "suf4Prev", "pre4Next=", "pre4Prev=", "pre4=", "suf5Next=", "headForm", "suf5Prev=", "suf5=", "shapeTrigram=", "pre3Next", "pre3Prev", "lemma=", "formlowerPrev", "formlowerNext", "pre4Next", "pre4Prev", "suf5Next", "suf5Prev", "lemsuf", "lemma"],
        orig_train_file=os.path.join(paths.PREPROCESS, "orig_train.feat"),
        orig_devel_file=os.path.join(paths.PREPROCESS, "orig_devel.feat"),
        orig_test_file=os.path.join(paths.PREPROCESS, "orig_test.feat"),
        full_train_file=os.path.join(paths.PREPROCESS, "train.feat"),
        full_devel_file=os.path.join(paths.PREPROCESS, "devel.feat"),
        full_test_file=os.path.join(paths.PREPROCESS, "test.feat")
    )
    print(results)

    """
    orig_train_file=os.path.join(paths.PREPROCESS, "orig_train.feat")
    orig_devel_file=os.path.join(paths.PREPROCESS, "orig_devel.feat")
    orig_test_file=os.path.join(paths.PREPROCESS, "orig_test.feat")
    full_train_file=os.path.join(paths.PREPROCESS, "train.feat")
    full_devel_file=os.path.join(paths.PREPROCESS, "devel.feat")
    full_test_file=os.path.join(paths.PREPROCESS, "test.feat")
    orig_train_data = read_features_file(orig_train_file)
    orig_devel_data = read_features_file(orig_devel_file)
    orig_test_data = read_features_file(orig_test_file)

    full_train_data = read_features_file(full_train_file)
    full_devel_data = read_features_file(full_devel_file)
    full_test_data = read_features_file(full_test_file)
    all_full_features = get_all_distinct_features_from_data(full_train_data)
    original_features = get_all_distinct_features_from_data(orig_train_data)
    prefixes = ["hasDot", "hasComma", "hasSlash", "hasPlus", "hasMinus", "isPunct", "isAscii", "isGreekLetter", "isCompound", "inNounChunk", "B-NP", "E-NP",  "hasInternalCap", "looksLikeFormula", "isInsideParentheses", "isParenAbbrev", "isMedicalModifier", "drugListStart", "possibleDiscontinuous", "external_in_dict", "externalpart_in_dict", "hasDash" ,"prevIsPunct", "NextIsPunct", "isDigit", "hasDigit", "isUpper", "isTitle", "isAllUpper", "isPlural", "NextIsLeftParen", "NextIsRightParen", "prevIsComma", "NextIsComma", "is_hyp_drug", "is_hyp_group", "isUnit","isAlpha", "nChildren", "prev_bigram_is_", "next_bigram_is_", "nearDict", "lenToken", "isStopword", "most_freq_class", "context", "in_missing_", "externalpart", "nearTrigger", "TopPre", "TopSuf", "tfidf_cosine_centroid", "headPos", "pos=", "childPos", "external", "headDep", "childDep", "dep=", "tag=", "cshape=", "shape=", "shapeBigramNext=", "suf3Next", "suf3Prev", "pre3Next=", "pre3Prev=", "pre3=", "suf4Next", "suf4Prev", "pre4Next=", "pre4Prev=", "pre4=", "suf5Next=", "headForm", "suf5Prev=", "suf5=", "shapeTrigram=", "pre3Next", "pre3Prev", "lemma=", "formlowerPrev", "formlowerNext", "pre4Next", "pre4Prev", "suf5Next", "suf5Prev", "lemsuf", "lemma"]
    candidate_features= get_features_with_prefixes(all_full_features, prefixes)
    candidate_features=candidate_features - original_features
    candidate_groups= get_feature_groups_by_prefix(candidate_features, prefixes)
    model_name = "LinearSVM"
    best_prefixes, selected_features, best_dev_score =select_best_features_for_model_greedy(model_name,candidate_groups,orig_train_data,full_train_data,full_devel_data,full_test_data)
    print(best_dev_score)
    print(best_prefixes)
    
    prefixes=["formlowerPrev", "formlowerNext", "lenToken", "hasDash", "hasDot", "hasComma", "hasSlash", "hasPlus", "hasMinus", "isPunct","prevIsPunct", "NextIsPunct","isAlpha", "isDigit", "hasDigit", "isAscii", "isGreekLetter", "isStopword", "pos=", "tag=", "dep=", "headForm", "headPos", "headDep", "nChildren", "childDep", "childPos", "isCompound", "inNounChunk", "B-NP", "E-NP", "most_freq_class", "prev_bigram_is_", "next_bigram_is_", "tfidf_cosine_centroid", "pre3=", "pre4=", "suf5=", "pre3Prev=", "pre4Prev=", "suf5Prev=", "pre3Next=", "pre4Next=", "suf5Next=", "pre3Prev", "pre4Prev", "suf3Prev", "suf4Prev", "suf5Prev", "pre3Next", "pre4Next", "suf3Next", "suf4Next", "suf5Next", "lemma=", "lemsuf", "lemma", "shape=", "cshape=", "isUpper", "isTitle", "isAllUpper", "hasInternalCap", "looksLikeFormula", "isUnit", "isPlural", "NextIsLeftParen", "NextIsRightParen", "prevIsComma", "NextIsComma", "isInsideParentheses", "shapeBigramNext=", "shapeTrigram=", "isParenAbbrev", "isMedicalModifier", "nearTrigger", "drugListStart", "possibleDiscontinuous", "is_hyp_drug", "is_hyp_group", "external", "externalpart", "external_in_dict", "externalpart_in_dict", "nearDict", "in_missing_", "TopPre", "TopSuf", "context"],
    model_name = "CRF"
    selected_train = os.path.join(paths.PREPROCESS, f"selected_train_{model_name}.feat")
    selected_devel = os.path.join(paths.PREPROCESS, f"selected_devel_{model_name}.feat")
    selected_test = os.path.join(paths.PREPROCESS, f"selected_test_{model_name}.feat")
    selected_features=None
    dict_ =train_final_model_and_evaluate("CRF", selected_train, selected_devel, selected_test, selected_features)
    print(dict_)
    """