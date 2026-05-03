import os
from evaluator import evaluate

def get_devel_out_files(directory):
    files = []
    for f in os.listdir(directory):
        if "devel" in f and f.endswith(".out"):
            files.append(f)
    return files

def get_test_out_files(directory):
    files = []
    for f in os.listdir(directory):
        if "test" in f and f.endswith(".out"):
            files.append(f)
    print("files", files)
    return files


DEVEL_XML = "/home/natalia/Escritorio/MAI/AHLT/labAHLT--1-/data/devel.xml"
TEST_XML = "/home/natalia/Escritorio/MAI/AHLT/labAHLT--1-/data/test.xml"
path = "/home/natalia/Escritorio/MAI/AHLT/labAHLT--1-/code/1.3.NERC-LLM/results"
files = get_devel_out_files(path)

for file in files:
    name = file.replace("-devel", "").replace(".out", "")
    devel_pred_file= path + "/" + file
    devel_stats_file= path + f"/devel-{name}.stats"
    evaluate("NER", DEVEL_XML, devel_pred_file, devel_stats_file)
