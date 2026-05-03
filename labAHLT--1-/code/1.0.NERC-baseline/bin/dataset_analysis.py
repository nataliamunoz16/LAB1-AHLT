import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_nerc_xml(file_path, split_name):
    if not os.path.exists(file_path): return None
    
    tree=ET.parse(file_path)
    root=tree.getroot()
    entity_data=[]
    
    for doc in root.findall('document'):
        for sentence in doc.findall('sentence'):
            for entity in sentence.findall('entity'):
                offset =entity.get('charOffset')
                try:
                    first_part=offset.split(';')[0]
                    start, end= map(int, first_part.split('-'))
                    length=end-start + 1
                except: length=0
                entity_data.append({'split':split_name,'type':entity.get('type'),'text':entity.get('text'),'char_length':length})
    return pd.DataFrame(entity_data)

# Load available splits
splits={'train':'data/train.xml','devel':'data/devel.xml','test':'data/test.xml'}
dfs=[parse_nerc_xml(path, name) for name, path in splits.items()]
full_df=pd.concat([d for d in dfs if d is not None])

# Figure 1: Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=full_df, x='type', hue='split', palette='Set2')
plt.title('NERC: entity frequency per split')
plt.savefig('nerc_distribution.png')

# Figure 2: Length analysis (violin)
plt.figure(figsize=(10, 6))
sns.violinplot(data=full_df, x='type', y='char_length', hue='split', inner="quart")
plt.title('Morphological variation: entity character lengths')
plt.savefig('nerc_morphology.png')