#! /usr/bin/python3

import sys, os
import re
from xml.dom.minidom import parse
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import paths
from dictionaries import Dictionaries
import unicodedata
from functions_for_extraction import extract_affix, extract_context_words, compute_cosine_tfidf, extract_tfidf, build_context_bigrams_dict, build_most_frequent_class_dict, drugbank_vs_train
from nltk.corpus import wordnet as wn


######################################
## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML
def get_label(tks, tke, spans) :
   for (spanS,spanE,spanT) in spans :
      if tks==spanS and tke<=spanE+1 : return "B-"+spanT
      elif tks>spanS and tke<=spanE+1 : return "I-"+spanT
   return "O"

def wordshape(token):
   """
   Converts a word into a template of characters and digits
   """
   out = []
   for ch in token:
      if ch.isupper():
         out.append("X")
      elif ch.islower():
         out.append("x")
      elif ch.isdigit():
         out.append("d")
      else:
         out.append(ch)
   return "".join(out)

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

def compact_shape(shape):
   """
   Colapses repetitions: XXXX --> X+
   """
   if not shape:
      return shape
   res = []
   prev = shape[0]
   run_len = 1
   for ch in shape[1:]:
      if ch == prev:
         run_len += 1
      else:
         res.append(prev+"+" if run_len > 1 else prev)
         prev = ch
         run_len = 1
   res.append(prev+"+" if run_len > 1 else prev)
   return "".join(res)


## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_sentence_features(tokens, dicts, affixes, words_prev_next, total_scores, words_special_train, freq_class, mostfreq_before_after) :

   
   sentenceFeatures = {}
   #Regex to detect units
   unit_pattern = re.compile(r"^(mg|g|kg|mcg|µg|ml|l|dl|mmol|mol|iu|units)$", re.I)
   #Pattern to detect chemical (letters+digits then hypen then letters/digits/greek)
   chem_hyphen_re = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+-[A-Za-z0-9µα-ωΑ-Ω]+$") #Leters + '-' + num/letter
   #Letters and digits mixed
   alnum_mix_re = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+$")
   
   #Check if each token is inside parentesis, brackets or braces
   paren_depth = 0
   inside_parens = [False] * len(tokens)
   for i_parens, tk_parens in enumerate(tokens):
      inside_parens[i_parens] = (paren_depth > 0)
      if tk_parens.text in ("(", "[", "{"):
         paren_depth += 1
      elif tk_parens.text in (")", "]", "}"):
         paren_depth = max(0, paren_depth - 1)

   #Noun chunk spans
   chunk_spans = [(c.start, c.end) for c in tokens.noun_chunks]
   
   for i,tk in enumerate(tokens) :
      
      tokenFeatures = []
      t = tk.text
      #Surface form features
      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())
      #Preffix/suffix features
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])
      tokenFeatures.append("suf5="+t[-5:])
      tokenFeatures.append("pre3="+t[:3])
      tokenFeatures.append("pre4="+t[:4])
      #Orthographic flags
      if t.isupper() : tokenFeatures.append("isUpper")
      if t.istitle() : tokenFeatures.append("isTitle")
      if t.isdigit() : tokenFeatures.append("isDigit")
      if '-' in t : tokenFeatures.append("hasDash")
      if re.search('[0-9]',t) : tokenFeatures.append("hasDigit")

      #Dictionary match features and drug presence
      found,val = dicts.find(t.lower(), 'external')
      if found:
         for c in val : tokenFeatures.append("external="+c)
         tokenFeatures.append(f"external_in_dict=1")
      found,val = dicts.find(t.lower(), 'externalpart')
      if found:
         for c in val : tokenFeatures.append("externalpart="+c)
         tokenFeatures.append(f"externalpart_in_dict=1")

      # Missing entry in drug bank
      for category in ["missing_drug", "missing_drug_n", "missing_brand", "missing_group"]:
         # Check if the current token exists in the set for this category
         if t.lower() in words_special_train.get(category, set()):
            tokenFeatures.append(f"in_{category}=1")
         else:
            tokenFeatures.append(f"in_{category}=0")
      
      #Most frequent bigrams in entities context
      for ent_type, contexts in mostfreq_before_after.items():
         #Check anterior
         if i >= 2:
            prev_bigram = (tokens[i-2].text.lower(), tokens[i-1].text.lower())
            if prev_bigram in contexts.get('anterior', []):
               tokenFeatures.append(f"prev_bigram_is_{ent_type}_context=1")
         #Check posterior 
         if i < len(tokens) - 2:
            next_bigram = (tokens[i+1].text.lower(), tokens[i+2].text.lower())
            if next_bigram in contexts.get('posterior', []):
               tokenFeatures.append(f"next_bigram_is_{ent_type}_context=1")
      
      # Hypernyms
      synsets = wn.synsets(t.lower(), pos=wn.NOUN)
      is_hyp_drug = "0"
      is_hyp_group = "0"
      if synsets:
         #We look at the primary sense of the word
         #.closure(lambda s: s.hypernyms()) gets all ancestors in the tree
         all_ancestors = [anc.name().split('.')[0] for anc in synsets[0].closure(lambda s: s.hypernyms())]
         drug_triggers = {'substance', 'compound', 'chemical', 'molecule', 'element'}
         if any(trigger in all_ancestors for trigger in drug_triggers):
            is_hyp_drug = "1"
         group_triggers = {'agent', 'inhibitor', 'blocker', 'antagonist', 'activator', 'medicine'}
         if any(trigger in all_ancestors for trigger in group_triggers):
            is_hyp_group = "1"
      tokenFeatures.append(f"is_hyp_drug={is_hyp_drug}")
      tokenFeatures.append(f"is_hyp_group={is_hyp_group}")

      # Most frequent class for each entity
      found_mfc = "NONE"
      for length in [3, 2, 1]:
         if i + length <= len(tokens):
            #Have a window of tokens
            window = tuple([tokens[j].text.lower() for j in range(i, i + length)])
            
            if window in freq_class:
                  found_mfc = freq_class[window]
                  break#To find the longest possible match
      tokenFeatures.append("most_freq_class=" + found_mfc)

      #Upper word
      is_all_upper = "1" if tk.text.isupper() and len(tk.text) > 1 else "0"
      tokenFeatures.append(f"isAllUpper={is_all_upper}")

      #Plural
      is_plural = "1" if "Number=Plur" in tk.morph else "0"
      tokenFeatures.append(f"isPlural={is_plural}")
      
      #Context word features from training statistics
      prev_low = tokens[i-1].lower_ if i > 0 else None
      next_low = tokens[i+1].lower_ if i < len(tokens)-1 else None
      #If neighbor word belongs to a top list for some entity context add a feature
      for ent_type, context_ in words_prev_next.items():
         if prev_low and prev_low in context_["left_top"]:
            tokenFeatures.append(f"contextLeftTop={ent_type}")
         if next_low and next_low in context_["right_top"]:
            tokenFeatures.append(f"contextRightTop={ent_type}")

      #Affix features based on previous prefixes/suffixes of entities in the train
      tlow = t.lower()
      for n in (2, 3, 4):
         if len(tlow) >= n:
            pref = tlow[:n]
            suf  = tlow[-n:]
            for ent_type in ("drug", "group", "brand", "drug_n"):
                  if pref in affixes.get(f"{ent_type}.pref{n}", []):
                     tokenFeatures.append(f"TopPref{n}={ent_type}")
                  if suf in affixes.get(f"{ent_type}.suf{n}", []):
                     tokenFeatures.append(f"TopSuf{n}={ent_type}")
      #Lemma features
      lem = tk.lemma_.lower()
      tokenFeatures.append("lemma="+lem)
      tokenFeatures.append("lemsuf3="+lem[-3:])
      tokenFeatures.append("lemsuf4="+lem[-4:])

      #Parentheses + stopwords
      if inside_parens[i]:
         tokenFeatures.append("isInsideParentheses")
      if tk.lower_ in STOP_WORDS:
         tokenFeatures.append("isStopword")
      
      #Greek letter detection (unicode + common greek letters)
      try:
         if "GREEK" in unicodedata.name(tk.text):
            tokenFeatures.append("isGreekLetter")
      except:
         pass
      if tk.text.lower() in ["alpha","beta","gamma","delta"]:
         tokenFeatures.append("isGreekLetterName")

      #Shape templates and a compact version
      sh = wordshape(t)
      tokenFeatures.append("shape="+sh)
      tokenFeatures.append("cshape="+compact_shape(sh))

      #Internal capitalization, alphanumeric (nums + letters or letter + nums)
      if re.search(r'[A-Z]', t[1:]):tokenFeatures.append("hasInternalCap")
      if re.search(r'\d+[a-zA-Z]|[a-zA-Z]\d+', t): tokenFeatures.append("hasAlphaNum")
      #Has dot or comma
      if '.'in t:tokenFeatures.append("hasDot")
      if ','in t:tokenFeatures.append("hasComma")

      #Unit token
      if unit_pattern.match(tk.text):
         tokenFeatures.append("isUnit")
      
      #Slash and (+) and (-) patterns
      if "/" in t:
         tokenFeatures.append("hasSlash")
      if "(+)" in t:
         tokenFeatures.append("hasPlus")
      if "(-)" in t:
         tokenFeatures.append("hasMinus")
      #In case it tokenizes (+ ) or ( +)
      window = tokens[max(0, i-1):min(len(tokens), i+2)]
      w = "".join([x.text for x in window])
      if not "(+)" in t:
         if "(+)" in w:
            tokenFeatures.append("hasPlus")
      if not "(-)" in t:
         if "(-)" in w:
            tokenFeatures.append("hasMinus")
      
      #Simple formula pattern
      if re.search(r'\b[A-Z][a-z]?\d*\b', t):  
         tokenFeatures.append("looksLikeFormula")
      
      #Chemical looking
      if chem_hyphen_re.match(t):
         tokenFeatures.append("isAlphaNumHyphen")
      if alnum_mix_re.match(t):
         tokenFeatures.append("isAlphaNumMix")
      
      #Pos tags
      tokenFeatures.append("pos=" + tk.pos_)
      tokenFeatures.append("tag=" + tk.tag_)

      #Letters, ASCII, punctuation
      if tk.is_alpha:
         tokenFeatures.append("isAlpha")
      if tk.is_ascii:
         tokenFeatures.append("isAscii")
      if tk.is_punct:
         tokenFeatures.append("isPunct")

      #Dependency parsing features
      tokenFeatures.append("dep=" + tk.dep_)
      tokenFeatures.append("headForm=" + tk.head.text.lower())
      tokenFeatures.append("headPos=" + tk.head.pos_)
      tokenFeatures.append("headDep=" + tk.head.dep_)

      #Children info (limited to 2 children)
      children = list(tk.children)
      tokenFeatures.append("nChildren=" + str(min(len(children), 3)))
      for ch in children[:2]:
         tokenFeatures.append("childDep=" + ch.dep_)
         tokenFeatures.append("childPos=" + ch.pos_)
      
      #Compound dependency
      if tk.dep_ == "compound":
         tokenFeatures.append("isCompound")
      
      #Beginning and end markers for NP chunk
      in_np = any(s <= i < e for (s,e) in chunk_spans)
      if in_np:
         tokenFeatures.append("inNounChunk")
         for (s,e) in chunk_spans:
            if i == s: tokenFeatures.append("B-NP")
            if i == e-1: tokenFeatures.append("E-NP")
      
      #Abbreviation inside parenthesis
      if 0 < i < len(tokens)-1:
         if tokens[i-1].text == "(" and tokens[i+1].text == ")" and t.isupper() and 2 <= len(t) <= 6:
            tokenFeatures.append("isParenAbbrev")
      # In case it tokenizes (ABBR ) or ( ABBR ) or ( ABBR)
      core = t.strip("()")
      if 2 <= len(t) <= 6 and core.isupper():
         window = tokens[max(0, i-1):min(len(tokens), i+2)]
         w = "".join([x.text for x in window])
         if f"({core})" in w:
            tokenFeatures.append("isParenAbbrev")
      #Len token
      L = len(t)
      tokenFeatures.append("lenToken=" +
         ("1" if L==1 else
         "2-3" if L<=3 else
         "4-6" if L<=6 else
         "7-10" if L<=10 else
         "10+"))
      
      #Medical trigger in a +/-3 window
      triggers = {"dose","doses","therapy","treatment","treated","administered","administration","antibiotic","antagonist"}
      window = 3
      for j in range(max(0,i-window), min(len(tokens), i+window+1)):
         if j != i and tokens[j].lower_ in triggers:
            tokenFeatures.append("nearTrigger="+tokens[j].lower_)
            break
      
      #Medical modifiers
      if t.lower() in ['oral', 'only', 'daily', 'systemic']:
         tokenFeatures.append("isMedicalModifier")
      
      #If token is in external drug dict and the next is and/or (drug and/or drug)
      if i < len(tokens)-2:
         if tokens[i+1].lower_ in {"and","or"}:
            if dicts.find(tokens[i].lower_,'external')[0]:
                  tokenFeatures.append("drugListStart")
      
      #If the token is a generic head and there is a possible entity in the same range, it can be a discontinuous mention
      if tk.lower_ in {"therapy","treatment","dose"}:
         for j in range(max(0,i-5), i):
            if dicts.find(tokens[j].lower_,'external')[0]:
                  tokenFeatures.append("possibleDiscontinuous")
                  break
         for j in range(i+1, min(len(tokens), i+6)):
            if dicts.find(tokens[j].lower_,'external')[0]:
                  tokenFeatures.append("possibleDiscontinuous")
                  break
      
      #nearby possible entity from dictionary
      window = 3
      for j in range(max(0,i-window), min(len(tokens), i+window+1)):
         if j!=i:
            found,val = dicts.find(tokens[j].lower_,'external')
            if found:
                  for c in val : tokenFeatures.append("nearDict="+c)
                  break
      
      #Shape bigram and trigram
      cur_shape = tk.shape_
      if i < len(tokens)-1:
         next_shape = tokens[i+1].shape_
         tokenFeatures.append("shapeBigramNext=" + cur_shape + "_" + next_shape)
      if 0 < i < len(tokens)-1:
         tokenFeatures.append("shapeTrigram=" + tokens[i-1].shape_ + "_" + cur_shape + "_" + tokens[i+1].shape_)
      
      #POS bigram and trigram
      cur_pos = tk.pos_
      if i < len(tokens)-1:
         next_pos = tokens[i+1].pos_
         tokenFeatures.append("posBigramNext=" + cur_pos + "_" + next_pos)
      if 0 < i < len(tokens)-1:
         tokenFeatures.append("posTrigram=" + tokens[i-1].pos_ + "_" + cur_pos + "_" + tokens[i+1].pos_)
      
      #TF-IDF centroid cosine
      for k in total_scores.keys():
         value_cosine = intervals_cosine(total_scores[k][i])
         tokenFeatures.append("tfidf_cosine_centroid_"+k+str(value_cosine))

      #Previous token features
      if i>0 :
         tPrev = tokens[i-1].text
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         tokenFeatures.append("suf5Prev="+tPrev[-5:])
         tokenFeatures.append("pre3Prev="+tPrev[:3])
         tokenFeatures.append("pre4Prev="+tPrev[:4])
         if tPrev.isupper() : tokenFeatures.append("isUpperPrev")
         if tPrev.istitle() : tokenFeatures.append("isTitlePrev")
         if tPrev.isdigit() : tokenFeatures.append("isDigitPrev")
         if '-' in tPrev : tokenFeatures.append("hasDashPrev")
         if re.search('[0-9]',tPrev) : tokenFeatures.append("hasDigitPrev")
         
         #Dictionary matches on previous token
         found,val = dicts.find(tPrev.lower(), 'external')
         if found:
            for c in val : tokenFeatures.append("externalPrev="+c)
         found,val = dicts.find(tPrev.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartPrev="+c)
         #Lemma features for previous token
         lem = tokens[i-1].lemma_.lower()
         tokenFeatures.append("lemmaPrev="+lem)
         tokenFeatures.append("lemsuf3Prev="+lem[-3:])
         tokenFeatures.append("lemsuf4Prev="+lem[-4:])

         #Punctuation/parenthesis/comma flags
         if tokens[i-1].is_punct:
            tokenFeatures.append("prevIsPunct")
         if tPrev == "(":
            tokenFeatures.append("prevIsLeftParen")
         if tPrev == ")":
            tokenFeatures.append("prevIsRightParen")
         if tPrev == ",":
            tokenFeatures.append("prevIsComma")

         #Stopwords and units
         if tokens[i-1].lower_ in STOP_WORDS:
            tokenFeatures.append("isStopwordPrev")
         if unit_pattern.match(tPrev):
            tokenFeatures.append("isUnitPrev")
      else :
         tokenFeatures.append("BoS")

      #Next token features
      if i<len(tokens)-1 :
         tNext = tokens[i+1].text
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         tokenFeatures.append("suf5Next="+tNext[-5:])
         tokenFeatures.append("pre3Next="+tNext[:3])
         tokenFeatures.append("pre4Next="+tNext[:4])
         if tNext.isupper() : tokenFeatures.append("isUpperNext")
         if tNext.istitle() : tokenFeatures.append("isTitleNext")
         if tNext.isdigit() : tokenFeatures.append("isDigitNext")
         if '-' in tNext : tokenFeatures.append("hasDashNext")
         if re.search('[0-9]',tNext) : tokenFeatures.append("hasDigitNext")
         
         #Dictionary matches on next token
         found,val = dicts.find(tNext.lower(), 'external')
         if found:
            for c in val : tokenFeatures.append("externalNext="+c)
         found,val = dicts.find(tNext.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartNext="+c)
         #Lemma features for next token
         lem = tokens[i+1].lemma_.lower()
         tokenFeatures.append("lemmaNext="+lem)
         tokenFeatures.append("lemsuf3Next="+lem[-3:])
         tokenFeatures.append("lemsuf4Next="+lem[-4:])
         #Punctuation/parenthesis/comma flags
         if tokens[i+1].is_punct:
            tokenFeatures.append("NextIsPunct")
         if tNext == "(":
            tokenFeatures.append("NextIsLeftParen")
         if tNext == ")":
            tokenFeatures.append("NextIsRightParen")
         if tNext == ",":
            tokenFeatures.append("NextIsComma")
         #Stopwords and units
         if tokens[i+1].lower_ in STOP_WORDS:
            tokenFeatures.append("isStopwordNext")
         if unit_pattern.match(tNext):
            tokenFeatures.append("isUnitNext")

      else:
         tokenFeatures.append("EoS")
      
      
      #Adds a richer left context
      if i>1 :
         tPrev = tokens[i-2].text
         tokenFeatures.append("formPrev2="+tPrev)
         tokenFeatures.append("formlowerPrev2="+tPrev.lower())
         tokenFeatures.append("suf3Prev2="+tPrev[-3:])
         tokenFeatures.append("suf4Prev2="+tPrev[-4:])
         tokenFeatures.append("suf5Prev2="+tPrev[-5:])
         tokenFeatures.append("pre3Prev2="+tPrev[:3])
         tokenFeatures.append("pre4Prev2="+tPrev[:4])
         if tPrev.isupper() : tokenFeatures.append("isUpperPrev2")
         if tPrev.istitle() : tokenFeatures.append("isTitlePrev2")
         if tPrev.isdigit() : tokenFeatures.append("isDigitPrev2")
         if '-' in tPrev : tokenFeatures.append("hasDashPrev2")
         if re.search('[0-9]',tPrev) : tokenFeatures.append("hasDigitPrev2")
         
         #Dictionary matches on previous-2 token
         found,val = dicts.find(tPrev.lower(), 'external')
         if found:
            for c in val : tokenFeatures.append("externalPrev2="+c)
         found,val = dicts.find(tPrev.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartPrev2="+c)
         #Lemma features for previous-2 token
         lem = tokens[i-2].lemma_.lower()
         tokenFeatures.append("lemmaPrev2="+lem)
         tokenFeatures.append("lemsuf3Prev2="+lem[-3:])
         tokenFeatures.append("lemsuf4Prev2="+lem[-4:])
         #Punctuation/parenthesis/comma flags
         if tokens[i-2].is_punct:
            tokenFeatures.append("prevIsPunct2")
         if tPrev == "(":
            tokenFeatures.append("prevIsLeftParen2")
         if tPrev == ")":
            tokenFeatures.append("prevIsRightParen2")
         if tPrev == ",":
            tokenFeatures.append("prevIsComma2")
         #Stopwords
         if tokens[i-2].lower_ in STOP_WORDS:
            tokenFeatures.append("isStopwordPrev2")
      else :
         tokenFeatures.append("BoS2")

      #Adds a richer right context
      if i<len(tokens)-2 :
         tNext = tokens[i+2].text
         tokenFeatures.append("formNext2="+tNext)
         tokenFeatures.append("formlowerNext2="+tNext.lower())
         tokenFeatures.append("suf3Next2="+tNext[-3:])
         tokenFeatures.append("suf4Next2="+tNext[-4:])
         tokenFeatures.append("suf5Next2="+tNext[-5:])
         tokenFeatures.append("pre3Next2="+tNext[:3])
         tokenFeatures.append("pre4Next2="+tNext[:4])
         if tNext.isupper() : tokenFeatures.append("isUpperNext2")
         if tNext.istitle() : tokenFeatures.append("isTitleNext2")
         if tNext.isdigit() : tokenFeatures.append("isDigitNext2")
         if '-' in tNext : tokenFeatures.append("hasDashNext2")
         if re.search('[0-9]',tNext) : tokenFeatures.append("hasDigitNext2")
         
         #Dictionary matches on next-2 token
         found,val = dicts.find(tNext.lower(), 'external')
         if found:
            for c in val : tokenFeatures.append("externalNext2="+c)
         found,val = dicts.find(tNext.lower(), 'externalpart')
         if found:
            for c in val : tokenFeatures.append("externalpartNext2="+c)
         #Lemma features for next-2 token
         lem = tokens[i+2].lemma_.lower()
         tokenFeatures.append("lemmaNext2="+lem)
         tokenFeatures.append("lemsuf3Next2="+lem[-3:])
         tokenFeatures.append("lemsuf4Next2="+lem[-4:])
         #Punctuation/parenthesis/comma flags
         if tokens[i+2].is_punct:
            tokenFeatures.append("NextIsPunct2")
         if tNext == "(":
            tokenFeatures.append("NextIsLeftParen2")
         if tNext == ")":
            tokenFeatures.append("NextIsRightParen2")
         if tNext == ",":
            tokenFeatures.append("NextIsComma2")
         #Stopwords
         if tokens[i+2].lower_ in STOP_WORDS:
            tokenFeatures.append("isStopwordNext2")
      else:
         tokenFeatures.append("EoS2")

      sentenceFeatures[i] = tokenFeatures
   return sentenceFeatures

## --------- Feature extractor ----------- 
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(datafile, outfile) :

   # load dictionaries
   dicts = Dictionaries(os.path.join(paths.RESOURCES,"dictionaries.json"))
   affixes = extract_affix(os.path.join("..","..", "..","data", "train.xml"), topk=7)
   words_prev_next = extract_context_words(os.path.join("..","..", "..","data", "train.xml"), topk=7)
   words_special_train = drugbank_vs_train(os.path.join("..","..", "..","resources", "DrugBank.txt"),os.path.join("..","..", "..","data", "train.xml"))
   freq_class=build_most_frequent_class_dict(os.path.join("..","..", "..","data", "train.xml"))
   mostfreq_before_after=build_context_bigrams_dict(os.path.join("..","..", "..","data", "train.xml"))
   num_context = 3
   vectorizer, centroids = extract_tfidf(os.path.join("..","..", "..","data", "train.xml"), num_context)
   total_scores = compute_cosine_tfidf(datafile, vectorizer=vectorizer, centroids=centroids, num_context= num_context)
   for ent_type in words_prev_next:
      words_prev_next[ent_type]["left_top"]=set(words_prev_next[ent_type]["left_top"])
      words_prev_next[ent_type]["right_top"]=set(words_prev_next[ent_type]["right_top"])
   # open output file
   outf = open(outfile, "w")
   
   # nlp = spacy.load("en_core_web_trf", enable=["tokenizer"])
   nlp = spacy.load("en_core_web_sm", disable=["ner"])
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datafile)

   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for j,s in enumerate(sentences) :
      sid = s.attributes["id"].value   # get sentence id
      print(f"extracting sentence {sid}        \r", end="")
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity") # get gold standard entities
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))

      # convert the sentence to a list of tokens
      tokens = nlp(stext)
      # extract sentence features
      scores_sentence = {k: v[j] for k, v in total_scores.items()}
      features = extract_sentence_features(tokens, dicts, affixes, words_prev_next, scores_sentence, words_special_train, freq_class, mostfreq_before_after)

      # print features in format expected by CRF/SVM/MEM trainers
      for i,tk in enumerate(tokens) :
         # see if the token is part of an entity
         tks,tke = tk.idx, tk.idx+len(tk.text)
         # get gold standard tag for this token
         tag = get_label(tks, tke, spans)
         # print feature vector for this token
         print (sid, tk.text, tks, tke-1, tag, "\t".join(features[i]), sep='\t', file=outf)

      # blank line to separate sentences
      print(file=outf)

   # close output file
   outf.close()

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__" :
   # directory with files to process
   datafile = sys.argv[1]
   # file where to store results
   featfile = sys.argv[2]
   
   extract_features(datafile, featfile)

