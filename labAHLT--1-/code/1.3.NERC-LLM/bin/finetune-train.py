import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import paths
from model import FineTuning
from examples import Examples
from prompts import Prompts

OTHERSCORE_NAMES={0: "random", 1: "score1", 2:"score2", 3:"perplexity"}
OTHERSCORES_=[0]
NUM_EXAMPLES=[-1]
LR_LIST=[1e-3]
BATCH_SIZE_LIST = [1]
EPOCHS = 10

def parse_csv_values(text, cast_fn):
    return [cast_fn(x.strip()) for x in text.split(",") if x.strip()]

# ------------ check command line and get arguments -----------------
def get_arguments():
    if not 5<=len(sys.argv)<=6  or (len(sys.argv)==6 and sys.argv[5]!="-quant"):
        print(f"Usage:  {sys.argv[0]} model prompts trainfile valfile [-quant]", file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    promptfile = sys.argv[2]
    traindata = sys.argv[3]
    valdata = sys.argv[4]
    quantized = (len(sys.argv)==6)

    return model, promptfile, traindata, valdata, quantized


def safe_name(text):
    return str(text).replace("/", "_")

def build_output_name(model, quantized, num_examples, lr, batch_size, otherscores):
    quant = "-quant" if quantized else ""
    score_name = OTHERSCORE_NAMES.get(otherscores, f"score{otherscores}")
    return (f"FT-{safe_name(model)}{quant}-n{num_examples}-{score_name}-lr{lr}-bs{batch_size}-ep{EPOCHS}.weights")
def select_train_examples(trainfile, num_examples, otherscores):
    examples= Examples(trainfile, "NER")
    if num_examples== -1:
        return examples.select_examples()
    examples = examples.select_examples(num_examples, otherscores=otherscores)
    print("EXAMPLES: ", len(examples))
    return examples

def run_one_experiment(model, quantized, prompts, train_examples, val_examples, lr, batch_size, num_examples, otherscores):
    model_path= model
    outputdir= os.path.join(paths.MODELS,build_output_name(model, quantized, num_examples, lr, batch_size, otherscores))
    os.makedirs(paths.MODELS, exist_ok=True)
    print(f"\nRUN: num_examples={num_examples} lr={lr} batch_size={batch_size} epochs={EPOCHS} quantized={quantized}",file=sys.stderr,flush=True)
    t0= time.time()
    engine= FineTuning(model_path, quantized=quantized)
    train_dataset = engine.tokenize_dataset(train_examples, prompts)
    val_dataset = engine.tokenize_dataset(val_examples, prompts)
    engine.train(train_dataset=train_dataset,val_dataset=val_dataset,outputdir=outputdir,learning_rate=lr,per_device_train_batch_size=batch_size,num_train_epochs=EPOCHS)
    elapsed= time.time()-t0
    print(f"Training took {elapsed:.1f} seconds", file=sys.stderr)
    result = {"outputdir": outputdir,"num_examples": num_examples,"otherscores": otherscores,"lr": lr,"batch_size": batch_size,"epochs": EPOCHS,"quantized": quantized,"train_size": len(train_examples),"val_size": len(val_examples),"elapsed_seconds": elapsed}
    with open(os.path.join(outputdir, "run_summary.json"), "w") as of:
        json.dump(result, of, indent=2)
    del engine
    torch.cuda.empty_cache()
    return result
############## MAIN ################

# get command line arguments
model, promptfile, traindata, valdata, quantized = get_arguments()
print(f"========= FINE TUNE == MODEL={model}  quantized={quantized}", file=sys.stderr)

# load prompts
prompts = Prompts(promptfile)
trainfile= os.path.join(paths.DATA, traindata + ".xml")
valfile= os.path.join(paths.DATA, valdata + ".xml")
val_examples = Examples(valfile, "NER").select_examples()
results = []
for otherscores in OTHERSCORES_:
    for num_examples in NUM_EXAMPLES:
        train_examples = select_train_examples(trainfile, num_examples, otherscores)
        for lr in LR_LIST:
            for batch_size in BATCH_SIZE_LIST:
                result = run_one_experiment(model=model,quantized=quantized,prompts=prompts,train_examples=train_examples,val_examples=val_examples,lr=lr,batch_size=batch_size,num_examples=num_examples, otherscores=otherscores)
                results.append(result)
print("Fine-tuning completed", file=sys.stderr)


# # load model and tokenizer
# t0 = time.time()
# MODEL_PATH = f"/scratch/nas/1/PDI/mml0/models/{model}"
# engine = FineTuning(MODEL_PATH, quantized=quantized)
# print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# # load and tokenize datasets
# t0 = time.time()
# trainfile = os.path.join(paths.DATA,traindata+".xml")
# train_examples = Examples(trainfile, "NER").select_examples()
# train_dataset = engine.tokenize_dataset(train_examples, prompts)

# valfile = os.path.join(paths.DATA,valdata+".xml")
# val_examples = Examples(valfile, "NER").select_examples()
# val_dataset = engine.tokenize_dataset(val_examples, prompts)
# print(f"Dataset loading took {time.time()-t0:.1f} seconds", file=sys.stderr)
        
# # Fine-tune the model and save results
# t0 = time.time()
# os.makedirs(paths.MODELS, exist_ok=True)
# quant="-quant" if quantized else ""
# outputdir = os.path.join(paths.MODELS, f"FT-{model}{quant}.weights")
# engine.train(train_dataset,
#              val_dataset, 
#              outputdir) 
# print(f"Training took {time.time()-t0:.1f} seconds", file=sys.stderr)

# print("Fine-tuning complete!", file=sys.stderr)

# # clean up gpu
# del engine
# torch.cuda.empty_cache() 

