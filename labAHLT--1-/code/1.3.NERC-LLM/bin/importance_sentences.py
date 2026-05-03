import nltk
import math
from collections import Counter
import xml.etree.ElementTree as ET
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


TYPES_ENTITIES=4
WEIGHTS = {"lexical_cross_entropy":0.3, "entity_types":0.25, "num_entities":0.2, "avg_idf": 0.15, "interesting_vocab_ratio":0.1}
NO_ENTITY_BONUS=0.05 #Small bonus to not penalize totally the sentences without entities
SIMILARITY_THRESHOLD=0.95 #Threshold to avoid having very similar sentences
STOPWORDS= stopwords.words('english')

def remove_sw(sentence):
    return [token.lower() for token in word_tokenize(sentence) if token.isalnum() and token.lower() not in STOPWORDS]

def parse_sentences_xml(xml_path):
    tree=ET.parse(xml_path)
    root=tree.getroot()
    sentences=[]
    for sentence in root.iter("sentence"):
        entities=[]
        for ent in sentence.iter("entity"):
            entities.append(ent.attrib.get("type"))
        text=sentence.attrib.get("text", "")
        sentences.append({"id": sentence.attrib.get("id"), "text": text, "tokens": remove_sw(text), "entities": entities})
    return sentences

def cross_entropy(tokens, probs):
    entropy=-sum(math.log2(probs[t]) for t in tokens if t in probs)
    return entropy/len(tokens) if len(tokens)>0 else 0

def compute_probs(sentences):
    probs={}
    all_tokens=[token for s in sentences for token in s["tokens"]]
    counts= Counter(all_tokens)
    total=sum(counts.values())
    probs={token: c/total for token,c in counts.items()}
    return probs

def compute_idf(sentences):
    N=len(sentences)
    df=Counter()
    for s in sentences:
        unique_tokens=set(s["tokens"])
        for token in unique_tokens:
            df[token]+=1
    idf={}
    for token,freq in df.items():
        idf[token]=math.log((N+1)/(freq+1))+1
    return idf

def cosine_similarity(vec1, vec2):
    dot= sum(a*b for a, b in zip(vec1, vec2))
    norm1= math.sqrt(sum(a*a for a in vec1))
    norm2= math.sqrt(sum(b*b for b in vec2))
    if norm1==0 or norm2==0:
        return 0.0
    return dot/(norm1*norm2)

def is_too_similar(candidate, selected):
    for chosen in selected:
        feat_sim= cosine_similarity(candidate["feature_vector"], chosen["feature_vector"])
        if feat_sim >= SIMILARITY_THRESHOLD:
            return True
    return False


def minmax_normalize(value, min_val, max_val):
    if max_val==min_val: return 0.5
    return (value-min_val)/(max_val-min_val)

def compute_sentence_scores(sentences):
    idf = compute_idf(sentences)
    probs=compute_probs(sentences)
    avg_global_idf= sum(idf.values()) /len(idf) if idf else 0
    for s in sentences:
        tokens=s["tokens"]
        entities=s["entities"]
        lexical_cross_entropy=cross_entropy(tokens, probs)
        entity_types=len(set(entities))/TYPES_ENTITIES
        num_entities=len(entities)
        token_idfs=[idf[t] for t in tokens if t in idf]
        avg_idf= sum(token_idfs)/len(token_idfs) if token_idfs else 0
        interesting_tokens= [t for t in tokens if idf.get(t, 0.0)>avg_global_idf]
        interesting_vocab_ratio=len(interesting_tokens)/len(tokens) if len(tokens)>0 else 0
        s["features_not_norm"]={
            "lexical_cross_entropy": lexical_cross_entropy,
            "entity_types": entity_types,
            "num_entities": num_entities,
            "avg_idf": avg_idf,
            "interesting_vocab_ratio":interesting_vocab_ratio,
            "no_entity_bonus": 0.05 if num_entities==0 else 0,
        }
    feature_names = s["features_not_norm"].keys()
    mins={}
    maxs={}
    for feat in feature_names:
        values=[s["features_not_norm"][feat] for s in sentences]
        mins[feat]=min(values)
        maxs[feat]=max(values)
    for s in sentences:
        s["features"]={
            "lexical_cross_entropy": minmax_normalize(s["features_not_norm"]["lexical_cross_entropy"],mins["lexical_cross_entropy"], maxs["lexical_cross_entropy"]),
            "entity_types": minmax_normalize(s["features_not_norm"]["entity_types"],mins["entity_types"], maxs["entity_types"]),
            "num_entities": minmax_normalize(s["features_not_norm"]["num_entities"],mins["num_entities"], maxs["num_entities"]),
            "avg_idf": minmax_normalize(s["features_not_norm"]["avg_idf"],mins["avg_idf"], maxs["avg_idf"]),
            "interesting_vocab_ratio":minmax_normalize(s["features_not_norm"]["interesting_vocab_ratio"],mins["interesting_vocab_ratio"], maxs["interesting_vocab_ratio"]),
            "no_entity_bonus": s["features_not_norm"]["no_entity_bonus"]}
        s["score"]= WEIGHTS["lexical_cross_entropy"]*s["features"]["lexical_cross_entropy"] + WEIGHTS["entity_types"]*s["features"]["entity_types"] + WEIGHTS["num_entities"]*(s["features"]["num_entities"]) + WEIGHTS["avg_idf"]*s["features"]["avg_idf"] + WEIGHTS["interesting_vocab_ratio"]*s["features"]["interesting_vocab_ratio"] + s["features"]["no_entity_bonus"]
        s["feature_vector"] = [s["features"]["lexical_cross_entropy"],s["features"]["entity_types"],s["features"]["num_entities"],s["features"]["avg_idf"],s["features"]["interesting_vocab_ratio"]]
    return sentences

def select_sentences_score2(sentences, top_k=50):
    ranked=sorted(sentences, key=lambda x:x["score"], reverse=True)
    selected = []
    for candidate in ranked:
        if not is_too_similar(candidate, selected):
            selected.append(candidate)
        if len(selected)>=top_k:
            break
    return selected

def select_top_k_perplexity(sentences, top_k=50):
    ranked= sorted(sentences,key=lambda s:2**s["features_not_norm"]["lexical_cross_entropy"],reverse=True)
    return ranked[:top_k]

