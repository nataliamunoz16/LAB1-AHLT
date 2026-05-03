import sys, random, re
from collections import Counter
from xml.dom.minidom import parse
from importance_sentences import select_sentences_score2, select_top_k_perplexity, parse_sentences_xml, compute_sentence_scores

class Examples() :

    # load dataset from xml file
    def __init__(self, xmlfile, task) :
       self.xmlfile=xmlfile
       self.task = task
       self.data = []
       tree = parse(xmlfile)   
       # process each sentence in the file
       sentences = tree.getElementsByTagName("sentence")
       for s in sentences :
          sid = s.attributes["id"].value   # get sentence id
          stext = s.attributes["text"].value.strip()   # get sentence text
          entities = s.getElementsByTagName("entity") # get gold standard entities
          spans = []
          for e in entities :
             # for discontinuous entities, we only get the first span
             # (will not work, but there are few of them)
             eid = e.attributes["id"].value
             (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
             typ =  e.attributes["type"].value
             spans.append((int(start),int(end),typ,eid)) 

          pairs = s.getElementsByTagName("pair") # get gold standard pairs
          interactions = []
          for p in pairs :          
             pid = s.attributes["id"].value   # get pair id
             # ground truth
             ddi = p.attributes["ddi"].value
             if (ddi=="true") : dditype = p.attributes["type"].value
             else : dditype = "null"
             # target entities
             e1 = p.attributes["e1"].value
             e2 = p.attributes["e2"].value
             interactions.append((e1,e2,dditype,pid))

          # start with the last span to simplfy offset computation
          spans.sort(reverse=True)
          if task == "NER":
              # rewrite expected output as XML-marked drugs in the text.
              newtext = stext
              for start,end,typ,_ in spans :
                  newtext = newtext[:end+1] + f"</{typ}>" + newtext[end+1:]
                  newtext = newtext[:start] + f"<{typ}>" + newtext[start:]

              self.data.append({"id" : sid,
                                "input" : stext,
                                "gold" : newtext
                               })
                               
          elif task == "DDI":
              for e1,e2,dditype,pid in interactions :
                  newtext = stext
                  for start,end,typ,eid in spans :
                      if eid==e1 : drug = "[DRUG1]"
                      elif eid==e2 : drug = "[DRUG2]"
                      else: drug = "[DRUG_OTHER]"
                      newtext = newtext[:start] + drug + newtext[end+1:]

                  self.data.append({"id": pid,
                                    "sid" : sid,
                                    "e1" : e1,
                                    "e2" : e2,
                                    "input" : newtext,
                                    "gold" : dditype
                                   })
                                   
          print(f"Loaded {len(self.data)} examples for {task} from {xmlfile}", file=sys.stderr)
          
                                     
    # ------------ select given number of FS examples -----------------
    def select_examples(self, numFS=-1, balanced=False, otherscores=2) :
        random.seed(12345)
        
        if balanced and self.task == "NER" :
            print(f"'balanced' not applicable for NER. Ignored", file=sys.stderr)
            balanced = False
        
        if numFS == -1 : 
            # return all dataset
            return self.data
            
        elif numFS == 0 : 
            # for zero shot
            return []
        elif otherscores == 1 and self.task == "NER":
            scored_data = []
            for ex in self.data:
                score = 0
                entities = re.findall(r'<(.*?)>(.*?)</\1>', ex['gold'])
                entity_count = len(entities)
                #various entities
                score += min(entity_count, 3) * 10 
                
                #complex (multi-word) entities
                multi_word = [e for e in entities if " " in e[1]]
                if multi_word:
                    score += 15
                
                #negative cases (no entities)
                if entity_count == 0:
                    score += 5 
                    
                #sentence length (short:<10,med:10-25,long:>25 words)
                words = ex['input'].split()
                if 10 < len(words) < 30: #prioritize medium
                    score += 10
                
                #position (beginning/middle/end)
                if ex['gold'].startswith('<'): score += 5
                if ex['gold'].strip().endswith('>'): score += 5

                scored_data.append({'ex': ex, 'score': score, 'text': ex['input']})

            #sort by score
            scored_data.sort(key=lambda x: x['score'], reverse=True)

            #select
            selected = []
            selected_texts = []
            
            for item in scored_data:
                if len(selected) >= numFS:
                    break
                
                #to avoid redundant examples
                is_redundant = False
                current_words = set(item['text'].lower().split())
                for seen_text in selected_texts:
                    seen_words = set(seen_text.lower().split())
                    intersection = current_words.intersection(seen_words)
                    union = current_words.union(seen_words)
                    if len(intersection) / len(union) > 0.4: # 40% overlap threshold
                        is_redundant = True
                        break
                
                if not is_redundant:
                    selected.append(item['ex'])
                    selected_texts.append(item['text'])

            print(f"Selected {len(selected)} informative examples", file=sys.stderr)
            return selected
        elif otherscores == 2 and self.task == "NER":
            sentences= parse_sentences_xml(self.xmlfile)
            sentences= compute_sentence_scores(sentences)
            selected_ranked= select_sentences_score2(sentences, top_k=numFS)
            selected_ids= {s["id"] for s in selected_ranked}
            selected_examples= [ex for ex in self.data if ex["id"] in selected_ids]
            print(f"Selected {len(selected_examples)} score2 examples", file=sys.stderr)
            return selected_examples
        
        elif otherscores == 3 and self.task == "NER":
            sentences= parse_sentences_xml(self.xmlfile)
            sentences= compute_sentence_scores(sentences)
            selected_ranked= select_top_k_perplexity(sentences, top_k=numFS)
            selected_ids= {s["id"] for s in selected_ranked}
            selected_examples= [ex for ex in self.data if ex["id"] in selected_ids]
            print(f"Selected {len(selected_examples)} perplexity examples", file=sys.stderr)
            return selected_examples

        elif balanced :               
            # return same amount of each class
            # only makes sense if there are few "gold" values (e.g. for DDI)

            # frequencies of each type
            types = Counter([x["gold"] for x in self.data])
            # balanced amout to expect
            n = max(1,numFS//len(types))        

            # start with less frequent
            examples = []
            pending = len(types)
            for t in sorted(types, key=lambda x : types[x]) :
                filtered = [x for x in self.data if x["gold"]==t]
                if len(filtered) < n :
                    # not enough examples, add them all
                    examples.extend(filtered)
                    # distribute the missing among the classes not yet processed
                    pending -= 1
                    n += (n-len(filtered)+1)//pending
                    print(f"Selected {len(filtered)} examples for class {t}", file=sys.stderr)
                else :
                    examples.extend(random.sample(filtered, n))
                    print(f"Selected {n} examples for class {t}", file=sys.stderr)

            random.shuffle(examples)
            print(f"Selected {len(examples)} balanced examples", file=sys.stderr)
            return examples

        else:
            # return random selection
            print(f"Selected {numFS} random examples", file=sys.stderr)
            return random.sample(self.data, numFS)
            
    def select_informative_examples(self, numFS):
        if numFS <= 0: return []
        
        scored_data = []
        for ex in self.data:
            score = 0
            entities = re.findall(r'<(.*?)>(.*?)</\1>', ex['gold'])
            entity_count = len(entities)
            #various entities
            score += min(entity_count, 3) * 10 
            
            #complex (multi-word) entities
            multi_word = [e for e in entities if " " in e[1]]
            if multi_word:
                score += 15
            
            #negative cases (no entities)
            if entity_count == 0:
                score += 5 
                
            #sentence length (short:<10,med:10-25,long:>25 words)
            words = ex['input'].split()
            if 10 < len(words) < 30: #prioritize medium
                score += 10
            
            #position (beginning/middle/end)
            if ex['gold'].startswith('<'): score += 5
            if ex['gold'].strip().endswith('>'): score += 5

            scored_data.append({'ex': ex, 'score': score, 'text': ex['input']})

        #sort by score
        scored_data.sort(key=lambda x: x['score'], reverse=True)

        #select
        selected = []
        selected_texts = []
        
        for item in scored_data:
            if len(selected) >= numFS:
                break
            
            #to avoid redundant examples
            is_redundant = False
            current_words = set(item['text'].lower().split())
            for seen_text in selected_texts:
                seen_words = set(seen_text.lower().split())
                intersection = current_words.intersection(seen_words)
                union = current_words.union(seen_words)
                if len(intersection) / len(union) > 0.4: # 40% overlap threshold
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(item['ex'])
                selected_texts.append(item['text'])

        print(f"Selected {len(selected)} informative examples", file=sys.stderr)
        return selected
    
    # ------------ convert xml marks to format expected by the evaluator ----------------
    def NER_eval_format(self, ex, text):
       xmlopen = "<drug>|<group>|<brand>|<drug_n>"
       
       original = text
       fmt = []
       p = re.search(xmlopen, text)
       while p is not None :
           os, oe = p.span()
           tag = p.group()[1:-1] # remove < >
           openlen = oe-os

           p = re.search(f"</{tag}>", text)  # find closing tag 

           ok = False
           if p is not None:
              cs, ce = p.span()       
              if f"<{tag}>" not in text[oe:cs]:
                 # make sure closing tag is the closest
                 drug = text[oe:cs]
                 ok = True
                 
           if not ok :
              print(f"Missing closing {tag} in: {original}", file=sys.stderr)
              
           text = text[:os]+text[oe:]  # remove opening mark
           if ok:
              cs -= openlen
              ce -= openlen
              text = text[:cs]+text[ce:]  # remove closing mark
              fmt.append(f"{ex['id']}|{os}-{cs-1}|{drug}|{tag}")
              
           p = re.search(xmlopen, text)
           
       return fmt       

    # ------------ convert example to format expected by the evaluator ----------------
    def DDI_eval_format(self, ex, text):
       p = text.find("\n")
       if p>0: text = text[:p]
       if text not in ["null", "none"] :
          return f"{ex['sid']}|{ex['e1']}|{ex['e2']}|{text}" 
       else:
          return ""



    # ------------ convert example to format expected by the evaluator ----------------
    def eval_format(self, ex, text):
        if self.task == "NER":
            return self.NER_eval_format(ex,text)
        elif self.task == "DDI":
            return self.DDI_eval_format(ex,text)

