from xml.dom.minidom import parse
from collections import defaultdict, Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np

# Sizes of prefixes and suffixes that will be extracted from the entities
SIZES = (2, 3, 4)

def intervals_cosine(score):
   """
   Discretize a cosine similarity into 6 bins
   """
   if (score < -0.6) and (score>=-1):
      return 0
   elif score < -0.2:
      return 1
   elif score < 0.1:
      return 2
   elif score < 0.4:
      return 3
   elif score < 0.6:
      return 4
   elif (score>=0.6) and (score<=1):
      return 5
   else:
      return 6

def add_affixes(dict_aff, ent_type, ent_text):
    """
    Adds prefix and suffix counts for a given entity text
    """
    ent_text = ent_text.strip().lower()
    for size in SIZES:
        #Skip if the entity is shorter than the affix size
        if len(ent_text)<size:
            continue
        pref = ent_text[:size]
        suf = ent_text[-size:]
        # Count the preffix/suffix frequency for this entity type
        dict_aff[f"{ent_type}.pref{size}"][pref] += 1
        dict_aff[f"{ent_type}.suf{size}"][suf] += 1
    return dict_aff

def extract_affix(xml_path, topk=7):
    """"
    Extract the most common prefixes and suffixes for each entity type
    """
    dict_ = defaultdict(Counter)
    #Parse XML file
    dom = parse(xml_path)
    entities = dom.getElementsByTagName("entity")
    #Collect the prefixxes and sufixxes for each entity
    for e in entities:
        ent_type = e.getAttribute("type").strip().lower()
        ent_text = e.getAttribute("text")
        if ent_text:
            dict_ = add_affixes(dict_, ent_type, ent_text)
    #Keep the top-k
    result = {}
    for key, counter in dict_.items():
        top = counter.most_common(topk)
        result[key] = [aff for aff, _ in top]
    return result

def good_tok(tok):
    """
    Filters the tokens and only accepts the ones that are not spages, punctuation nor stopwords
    """
    return (not tok.is_punct) and(not tok.is_space)and (tok.lower_ not in STOP_WORDS)

def extract_context_words(xml_path, topk):
    """
    Extracts most frequent context words appearing immediately before and after the entities
    """
    nlp = spacy.load("en_core_web_trf", disable=["transformer"])
    left_counts= defaultdict(Counter)
    right_counts= defaultdict(Counter)
    # Parse the xml
    doc = parse(xml_path)
    sentences = doc.getElementsByTagName("sentence")
    for s in sentences:
        stext = s.getAttribute("text")
        if not stext:
            continue
        #Tokenize the text
        doc = nlp(stext)
        #For each entity
        for e in s.getElementsByTagName("entity"):
            #Estract the type and the offset of characters
            type_ = e.getAttribute("type").strip().lower()
            off0 = e.getAttribute("charOffset").split(";")[0]
            start_str, end_str = off0.split("-")
            start = int(start_str)
            end_excl = int(end_str) + 1
            prev_tok = None
            next_tok = None
            # Find the token before and after the entity
            for tok in doc:
                ts = tok.idx
                te = tok.idx + len(tok)
                if te <= start:
                    prev_tok = tok
                    continue
                if ts >= end_excl:
                    next_tok = tok
                    break
            if prev_tok and good_tok(prev_tok):
                left_counts[type_][prev_tok.lower_] += 1
            if next_tok and good_tok(next_tok):
                right_counts[type_][next_tok.lower_] += 1
    result = {}
    #Collect all entity types seen
    all_types = sorted(list(left_counts.keys()) + list(right_counts.keys()))
    all_types = list(set(all_types))
    #Keep the topk context words
    for t in all_types:
        result[t] = {"left_top":  [w for w, _ in left_counts[t].most_common(topk)],"right_top": [w for w, _ in right_counts[t].most_common(topk)]}
    return result
def extract_next(doc, end):
    """
    Returns the next valid token (not punctuation or space)
    """
    next_tok = None
    te_next = None
    for tok in doc:
        if (not tok.is_punct) and (not tok.is_space):
            ts = tok.idx
            te = tok.idx + len(tok)
            if ts >= end:
                next_tok = tok
                te_next = te
                break
    return next_tok, te_next

def extract_prev(doc, start):
    """
    Returns the previous valid token (not punctuation or space)
    """
    prev_tok = None
    ts_prev = None
    for tok in doc:
        if (not tok.is_punct) and (not tok.is_space):
            ts = tok.idx
            te = tok.idx + len(tok)
            if te <= start:
                prev_tok = tok
                ts_prev = ts
                continue
    return prev_tok, ts_prev

def extract_n_context_entity(doc, off0, entity_text, n=3):
    """
    Extracts a context window of size n arround a word
    """
    start_str, end_str = off0.split("-")
    start = int(start_str)
    end_excl = int(end_str) + 1
    context = [entity_text.lower()]
    for i in range(n):
        prev_tok, ts_prev = extract_prev(doc, start)
        next_tok, te_next = extract_next(doc, end_excl)
        if prev_tok is not None:
            context.insert(0, prev_tok.text.lower())
        if next_tok is not None:
            context.append(next_tok.text.lower())
        if ts_prev is not None:
            start = ts_prev
        if te_next is not None:
            end_excl = te_next + 1
        if ts_prev is None and te_next is None:
            break
    # Remove all possible None values
    context = [x for x in context if x is not None]
    return context

def compute_centroids(X_train, labels_train):
    """
    Computes a centroid vector for each class using the TF-IDF vectors
    """
    centroids = {}
    X_train = normalize(X_train, norm="l2")
    for label in set(labels_train):
        #Indices of documents belonging to the class
        idx_labels = [i for i,label_ in enumerate(labels_train) if label_ == label]
        #Mean vector of the class
        centroid = X_train[idx_labels].mean(axis=0)
        centroid = np.asarray(centroid)
        centroid = normalize(centroid, norm="l2")
        centroids[label] = centroid
    return centroids
    
def extract_tfidf(xml_path, num_context=3):
    """
    Trains a TF-IDF representation using entity contexts, so that each entity context is treated as a document
    """
    nlp = spacy.load("en_core_web_trf", disable=["transformer"])
    # Parse the xml
    dom = parse(xml_path)
    sentences = dom.getElementsByTagName("sentence")
    docs_train = []
    labels_train = []
    for s in sentences:
        stext = s.getAttribute("text")
        if not stext:
            continue
        #Tokenize the text
        doc = nlp(stext)
        #For each entity
        for e in s.getElementsByTagName("entity"):
            #Estract the type and the offset of characters
            ent_type = e.getAttribute("type").strip().lower()
            off0 = e.getAttribute("charOffset").split(";")[0]
            entity_text = e.getAttribute("text").strip().lower()
            context = extract_n_context_entity(doc, off0, entity_text, num_context)
            labels_train.append(ent_type)
            docs_train.append(" ".join(context))
    #Train the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(docs_train)
    X_train = vectorizer.transform(docs_train)
    centroids = compute_centroids(X_train, labels_train)
    return vectorizer, centroids

def cosine_to_centroids(x_vec, centroid_matrix, classes):
    """
    Compute cosine similarity between a TF-IDF vector and all centroids
    """
    sims= cosine_similarity(x_vec, centroid_matrix)[0]
    scores = {c: float(sims[i]) for i, c in enumerate(classes)}
    return scores

def compute_cosine_tfidf(xml_path, vectorizer=None, centroids=None, num_context=3):
    """
    Computes the cosine similarity between token contexts and each class centroid, obtaining a result per token and entity type
    """
    if (vectorizer is None) or (centroids is None):
        return {}
    centroid_matrix = np.vstack([centroids[c] for c in centroids.keys()])
    # centroid_matrix = sparse.vstack([centroids[c] for c in centroids.keys()])
    nlp = spacy.load("en_core_web_trf", disable=["transformer"])
    dom = parse(xml_path)
    sentences = dom.getElementsByTagName("sentence")
    total_scores = {key: [] for key in centroids.keys()}
    for s in sentences:
        stext = s.getAttribute("text")
        if not stext:
            for c in centroids.keys():
                total_scores[c].append([])
            continue
        #Tokenize the text
        scores_sentence = {key: [] for key in centroids.keys()}
        doc = nlp(stext)
        for token in doc:
            #Ignore punctuation and spaces
            if token.is_punct or token.is_space:
                for c in list(centroids.keys()):
                    scores_sentence[c].append(-3)
                continue
            start = token.idx
            end = token.idx + len(token) - 1
            #Build token context
            context = extract_n_context_entity(doc, f"{start}-{end}", token.text, num_context)
            tf_idf_context = vectorizer.transform([" ".join(context)])
            scores = cosine_to_centroids(tf_idf_context, centroid_matrix, centroids.keys())
            for c in list(centroids.keys()):
                scores_sentence[c].append(scores.get(c, -3))
            # for k, v in scores.items():
            #     scores_sentence[k].append(v)
        # Safety check
        n_tokens = len(doc)
        for c in list(centroids.keys()):
            if len(scores_sentence[c]) < n_tokens:
                scores_sentence[c].extend([-3] * (n_tokens - len(scores_sentence[c])))
            elif len(scores_sentence[c]) > n_tokens:
                scores_sentence[c] = scores_sentence[c][:n_tokens]
            total_scores[c].append(scores_sentence[c])
        # for key_ in total_scores.keys():
        #     total_scores[key_].append(scores_sentence[key_])
    return total_scores

def drugbank_vs_train(drugbank_path, train_xml_path):
    nlp = spacy.load("en_core_web_trf", disable=["transformer"])
    db_names = set()
    try:
        with open(drugbank_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    db_names.add(line.split('|')[0].strip().lower())
    except FileNotFoundError:
        print(f"Warning: {drugbank_path} not found.")
    results={"missing_drug": set(),"missing_drug_n": set(),"missing_brand": set(),"missing_group": set()}

    dom=parse(train_xml_path)
    entities= dom.getElementsByTagName("entity")

    for e in entities:
        text =e.getAttribute("text")
        ent_type =e.getAttribute("type").lower()
        
        if text.lower() not in db_names:
            # Determine the key
            key= f"missing_{ent_type}" 
            if key in results:
                #Split "tricyclic antidepressants" -> ["tricyclic", "antidepressants"]
                tokenized_ent = nlp(text)
                for token in tokenized_ent:
                    # Add individual tokens
                    results[key].add(token.text.lower())
    return results


def build_most_frequent_class_dict(train_xml_path):
    nlp = spacy.load("en_core_web_trf", disable=["transformer"])
    phrase_counts=defaultdict(lambda: defaultdict(int))
    
    dom =parse(train_xml_path)
    entities =dom.getElementsByTagName("entity")

    for e in entities:
        text =e.getAttribute("text").lower()
        ent_type=e.getAttribute("type").lower()
        #Tokenize and convert to a tuple: "insulin glargine" -> ("insulin", "glargine")
        tokens_tuple=tuple([tk.text for tk in nlp(text)])
        phrase_counts[tokens_tuple][ent_type] += 1
    #Final map:{("insulin", "glargine"):"drug" }
    phrase_mfc={phrase: max(types, key=types.get) for phrase, types in phrase_counts.items()}
    return phrase_mfc


def build_context_bigrams_dict(train_xml_path, top_k=10):
    # Format: bigram_counts['group']['anterior'] = Counter({('treatment', 'with'): 25, ...})
    bigram_counts = {
        etype: {'anterior': Counter(), 'posterior': Counter()} 
        for etype in ['drug', 'drug_n', 'brand', 'group']
    }

    dom =parse(train_xml_path)
    sentences = dom.getElementsByTagName("sentence")

    for s in sentences:
        stext = s.getAttribute("text")
        entities = s.getElementsByTagName("entity")
        
        # Use a simple split or your nlp tokenizer
        words = [w.lower().strip(".,()[]") for w in stext.split()]

        for e in entities:
            ent_text = e.getAttribute("text")
            ent_type = e.getAttribute("type").lower()
            ent_words = [w.lower().strip(".,()[]") for w in ent_text.split()]
            
            try:
                start_idx = words.index(ent_words[0])
                end_idx = start_idx + len(ent_words) - 1

                # 1. Anterior Bigram (Word-2, Word-1)
                if start_idx >= 2:
                    ant_bigram = (words[start_idx-2], words[start_idx-1])
                    bigram_counts[ent_type]['anterior'][ant_bigram] += 1
                
                # 2. Posterior Bigram (Word+1, Word+2)
                if end_idx<len(words)-2:
                    post_bigram=(words[end_idx+1], words[end_idx+2])
                    bigram_counts[ent_type]['posterior'][post_bigram] += 1
            except ValueError:
                continue

    # Filter for Top K
    top_bigrams={}
    for etype in bigram_counts:
        top_bigrams[etype]={'anterior':[bg for bg,_ in bigram_counts[etype]['anterior'].most_common(top_k)],'posterior':[bg for bg,_ in bigram_counts[etype]['posterior'].most_common(top_k)]}
    return top_bigrams

