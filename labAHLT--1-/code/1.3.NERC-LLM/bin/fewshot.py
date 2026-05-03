import os,sys,time,json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import paths
from model import Inference
from prompts import Prompts
from examples import Examples
from evaluator import evaluate

OTHERSCORES=2
print(OTHERSCORES)
if OTHERSCORES==1:
    SCORE_NAME= "score1"
elif OTHERSCORES==2:
    SCORE_NAME= "score2"
elif OTHERSCORES==3:
    SCORE_NAME= "perplexity"
else:
    SCORE_NAME= "no_score"

PROMPT="prompt02"

# ------------ check command line and get arguments -----------------
def get_arguments():
    if (not 6<=len(sys.argv)<=7):
        print(f"Usage:  {sys.argv[0]} model prompts num_few_shot trainfile testfile [(-quant|-ollama)]", file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    promptfile = sys.argv[2]
    num_few_shot = int(sys.argv[3])
    traindata = sys.argv[4]
    testdata = sys.argv[5]
    quantized = False
    ollama = False

    if len(sys.argv) == 7:
        if sys.argv[6] == "-quant":
            quantized = True
        elif sys.argv[6] == "-ollama":
            ollama = True
        else:
            print("Last argument must be -quant or -ollama", file=sys.stderr)
            sys.exit(1)

    return model, promptfile, num_few_shot, traindata, testdata, quantized, ollama


############## main ###################

# get command line arguments
model, promptfile, num_few_shot, traindata, testdata, quantized, ollama = get_arguments()

print(f"========= FEW SHOT === PROMPTS={promptfile}  SHOTS={num_few_shot}  DATA={testdata} quantized={quantized}", file=sys.stderr)

# load training data (FS examples)
trainfile = os.path.join(paths.DATA,traindata+".xml")
fs_examples = Examples(trainfile, "NER").select_examples(num_few_shot, otherscores=OTHERSCORES)

# load prompts, create few-shot prompt
prompts = Prompts(promptfile, fs_examples)

# load test data
testfile = os.path.join(paths.DATA,testdata+".xml")
test = Examples(testfile, "NER")

# load model and tokenizer
t0 = time.time()
if ollama:
   engine = Inference(model, ollama=True)
else :
   # model = "Featherless-Chat-Models/Mistral-7B-Instruct-v0.2"
   MODEL_PATH = model
   engine = Inference(MODEL_PATH, quantized=quantized)
print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# annotate each example in testdata
t0 = time.time()
annotated = []
for i,ex in enumerate(test.select_examples()):
    print(f"Processing example {i} - {ex['id']}", flush=True, file=sys.stderr)
    
    # create prompt for this example, adding it to FS prompt
    messages = prompts.prepare_messages(ex['input'])
    # call model to generate response 
    gen_text = engine.generate(messages)
    # store responses
    ex['predicted'] = gen_text
    ex['evaluator'] = test.eval_format(ex,gen_text)
    annotated.append(ex)

print("Done", file=sys.stderr)
print(f"Processed {len(annotated)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(annotated):.2f} sec/example)", file=sys.stderr)

os.makedirs(paths.RESULTS, exist_ok=True)
quant = "-quant" if quantized else ""
safe_model = model.replace("/", "_")
outfname = os.path.join(paths.RESULTS,
                        f"FS-{safe_model}-{SCORE_NAME}-{PROMPT}-{num_few_shot}-{testdata}{quant}")
with open(outfname+".json", "w") as of:  
   json.dump(annotated, of, indent=1, ensure_ascii=False)
with open(outfname+".out", "w") as of:  
   for e in annotated:
      if e["evaluator"]: 
          print("\n".join(e["evaluator"]), file=of)

name = outfname.replace("-devel", "").replace(".out", "")
pred_file= outfname+".out"
if testdata=="devel":
   stats_file= paths.RESULTS + f"/devel-{name}.stats"
else:
   stats_file= paths.RESULTS + f"/test-{name}.stats"
evaluate("NER", testfile, pred_file, stats_file)

# clean up gpu
del engine
torch.cuda.empty_cache() 

