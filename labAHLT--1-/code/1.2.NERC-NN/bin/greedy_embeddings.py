import os
from train import do_train
from predict import predict
from evaluator import evaluate
from functions_for_extraction import extract_affix, extract_tfidf, compute_cosine_tfidf

THISDIR= os.path.abspath(os.path.dirname(__file__))
NERDIR= os.path.dirname(THISDIR)
SOLDIR= os.path.dirname(NERDIR)
MAINDIR= os.path.dirname(SOLDIR)
DATADIR= os.path.join(MAINDIR, "data")
TRAIN_PCK= os.path.join(NERDIR, "preprocessed", "train.pck")
DEVEL_PCK= os.path.join(NERDIR, "preprocessed", "devel.pck")
TRAIN_XML= os.path.join(DATADIR, "train.xml")
DEVEL_XML= os.path.join(DATADIR, "devel.xml")
TEST_PCK = os.path.join(NERDIR, "preprocessed", "test.pck")
TEST_XML = os.path.join(DATADIR, "test.xml")
MODELS_DIR = os.path.join(NERDIR, "models")
RESULTS_DIR = os.path.join(NERDIR, "results")

CANDIDATES = ["pref3", "pref5", "lemma", "pos"]
TRAIN_ARGS = {"epochs":10,"batch_size":16,"optimizer":"Adam","lr":0.001,"embLWsize":100,"embWsize":100,"embSsize":50,"dropout_rate":0.1,"lstm_out_size":200,"linear_out_size":200,"num_layers_lstm":1,"max_len":150,"suf_len":2,"dropout_lstm":True,"linear2":False,"seed":2345, "activation":"relu", "pretrained_emb_path":None, "step":[4]}

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


def build_resources():
    affixes = extract_affix(TRAIN_XML, topk=10)
    total_scores_train= None
    total_scores_devel= None
    vectorizer, centroids= extract_tfidf(TRAIN_XML, num_context=3)
    total_scores_train= compute_cosine_tfidf(TRAIN_XML, vectorizer=vectorizer, centroids=centroids, num_context=3)
    total_scores_devel= compute_cosine_tfidf(DEVEL_XML, vectorizer=vectorizer, centroids=centroids, num_context=3)
    total_scores_test = compute_cosine_tfidf(TEST_XML, vectorizer=vectorizer, centroids=centroids, num_context=3)
    return affixes, total_scores_train, total_scores_devel, total_scores_test
def flags_from_selected(selected):
    return {"use_pref3":"pref3" in selected,"use_pref5":"pref5" in selected,"use_lemma":"lemma" in selected,"use_pos":"pos" in selected}
def evaluate_embedding_set(selected):
    name= "greedy_embedding_" + ("baseline" if not selected else "_".join(map(str, selected)))
    model_dir= os.path.join(MODELS_DIR, name)
    pred_file= os.path.join(RESULTS_DIR, f"devel-{name}.out")
    stats_file= os.path.join(RESULTS_DIR, f"devel-{name}.stats")
    flags =flags_from_selected(selected)
    do_train(TRAIN_PCK,
            DEVEL_PCK,
            model_dir,
            epochs=TRAIN_ARGS["epochs"],
            batch_size=TRAIN_ARGS["batch_size"],
            optimizer=TRAIN_ARGS["optimizer"],
            lr=TRAIN_ARGS["lr"],
            embLWsize=TRAIN_ARGS["embLWsize"],
            embWsize=TRAIN_ARGS["embWsize"],
            embSsize=TRAIN_ARGS["embSsize"],
            dropout_rate=TRAIN_ARGS["dropout_rate"],
            lstm_out_size=TRAIN_ARGS["lstm_out_size"],
            linear_out_size=TRAIN_ARGS["linear_out_size"],
            num_layers_lstm=TRAIN_ARGS["num_layers_lstm"],
            max_len=TRAIN_ARGS["max_len"],
            suf_len=TRAIN_ARGS["suf_len"],
            dropout_lstm=TRAIN_ARGS["dropout_lstm"],
            linear2=TRAIN_ARGS["linear2"],
            step=TRAIN_ARGS["step"],
            affixes=AFFIXES,
            total_scores_train=TOTAL_SCORES_TRAIN,
            total_scores_val=TOTAL_SCORES_DEVEL,
            activation = TRAIN_ARGS["activation"],
            pretrained_emb_path=TRAIN_ARGS["pretrained_emb_path"],
            seed=TRAIN_ARGS["seed"],
            **flags)
    predict(model_dir,DEVEL_PCK,pred_file,batch_size=TRAIN_ARGS["batch_size"],max_len=TRAIN_ARGS["max_len"],suf_len=TRAIN_ARGS["suf_len"],step=TRAIN_ARGS["step"],affixes=AFFIXES,total_scores=TOTAL_SCORES_DEVEL)
    evaluate("NER", DEVEL_XML, pred_file, stats_file)
    f1= read_f1_from_stats(stats_file)
    return f1, name, stats_file

def greedy_forward_selection_embedding(candidates):
    selected= []
    remaining= candidates[:]
    best_f1, best_name, best_stats= evaluate_embedding_set([])
    print(f"\nBaseline F1={best_f1:.4f}")
    history = [([], best_f1, best_name, best_stats)]
    improved = True
    while improved and remaining:
        improved= False
        round_best_feature= None
        round_best_f1= best_f1
        round_best_name= None
        round_best_stats= None
        print(f"\nSelected: {selected}")
        print(f"Remaining: {remaining}")
        for feat in remaining:
            trial= selected + [feat]
            f1, name, stats_file = evaluate_embedding_set(trial)
            print(f"Trial {trial} F1={f1:.4f}")
            if f1>round_best_f1:
                round_best_f1=f1
                round_best_feature=feat
                round_best_name=name
                round_best_stats=stats_file
        if round_best_feature is not None:
            selected.append(round_best_feature)
            remaining.remove(round_best_feature)
            best_f1=round_best_f1
            best_name=round_best_name
            best_stats=round_best_stats
            history.append((selected[:], best_f1, best_name, best_stats))
            improved=True
            print(f"\n Feature added {round_best_feature}. New best F1={best_f1:.4f}")
        else:
            print("\nStop. None of the features improves the result")
    return selected, best_f1, history

def save_selected_greedy_outputs(selected):
    name= "selected_greedy_embeddings_"
    model_dir= os.path.join(MODELS_DIR, name)
    devel_pred_file= os.path.join(RESULTS_DIR, f"devel-{name}.out")
    devel_stats_file= os.path.join(RESULTS_DIR, f"devel-{name}.stats")
    test_pred_file= os.path.join(RESULTS_DIR, f"test-{name}.out")
    test_stats_file= os.path.join(RESULTS_DIR, f"test-{name}.stats")
    flags = flags_from_selected(selected)
    do_train(
        TRAIN_PCK,
        DEVEL_PCK,
        model_dir,
        epochs=TRAIN_ARGS["epochs"],
        batch_size=TRAIN_ARGS["batch_size"],
        optimizer=TRAIN_ARGS["optimizer"],
        lr=TRAIN_ARGS["lr"],
        embLWsize=TRAIN_ARGS["embLWsize"],
        embWsize=TRAIN_ARGS["embWsize"],
        embSsize=TRAIN_ARGS["embSsize"],
        dropout_rate=TRAIN_ARGS["dropout_rate"],
        lstm_out_size=TRAIN_ARGS["lstm_out_size"],
        linear_out_size=TRAIN_ARGS["linear_out_size"],
        num_layers_lstm=TRAIN_ARGS["num_layers_lstm"],
        max_len=TRAIN_ARGS["max_len"],
        suf_len=TRAIN_ARGS["suf_len"],
        dropout_lstm=TRAIN_ARGS["dropout_lstm"],
        linear2=TRAIN_ARGS["linear2"],
        step=TRAIN_ARGS["step"],
        affixes=AFFIXES,
        total_scores_train=TOTAL_SCORES_TRAIN,
        total_scores_val=TOTAL_SCORES_DEVEL,
        activation = TRAIN_ARGS["activation"],
        pretrained_emb_path=TRAIN_ARGS["pretrained_emb_path"],
        seed=TRAIN_ARGS["seed"],
        **flags)
    predict(
        model_dir,
        DEVEL_PCK,
        devel_pred_file,
        batch_size=TRAIN_ARGS["batch_size"],
        max_len=TRAIN_ARGS["max_len"],
        suf_len=TRAIN_ARGS["suf_len"],
        step=TRAIN_ARGS["step"],
        affixes=AFFIXES,
        total_scores=TOTAL_SCORES_DEVEL
    )
    evaluate("NER", DEVEL_XML, devel_pred_file, devel_stats_file)

    predict(
        model_dir,
        TEST_PCK,
        test_pred_file,
        batch_size=TRAIN_ARGS["batch_size"],
        max_len=TRAIN_ARGS["max_len"],
        suf_len=TRAIN_ARGS["suf_len"],
        step=TRAIN_ARGS["step"],
        affixes=AFFIXES,
        total_scores=TOTAL_SCORES_TEST
    )
    evaluate("NER", TEST_XML, test_pred_file, test_stats_file)
    return devel_pred_file, devel_stats_file, test_pred_file, test_stats_file

if __name__ == "__main__":
    AFFIXES, TOTAL_SCORES_TRAIN, TOTAL_SCORES_DEVEL, TOTAL_SCORES_TEST = build_resources()
    best_set, best_f1, history= greedy_forward_selection_embedding(CANDIDATES)
    print(f"Best set: {best_set}")
    print(f"Best F1: {best_f1:.4f}")
    print("History:")
    for feats, f1, name, stats in history:
        print(f"{feats} -> {f1:.4f} ({name})")
    
    devel_out, devel_stats, test_out, test_stats = save_selected_greedy_outputs(best_set)
    print("\nSaved final selected_greedy files:")
    print(f"DEVEL OUT: {devel_out}")
    print(f"DEVEL STATS: {devel_stats}")
    print(f"TEST OUT: {test_out}")
    print(f"TEST STATS: {test_stats}")