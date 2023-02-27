import spacy
import pandas as pd
from spacy.matcher import DependencyMatcher


def excel_to_df(filepath):
    '''
    [Extraction des données excel en df]
    [Spécifier:]
    [filepath = Localisation du fichier "data.csv"]
    '''
    try: 
        try:
            df = pd.read_csv(filepath)
            print('csv passed')
            return df
        except Exception as e:
            pass
        try:
            df = pd.read_excel(filepath)
            print('xlsx passed')
            return df
        except Exception as e:
            pass    
    except Exception as e:
        print('excel_to_df failed')
        print(f'error: {e}'.format(e))
        pass


nlp = spacy.load("en_core_web_sm")

pattern_1 = [
    {
      "RIGHT_ID": "target",
      "RIGHT_ATTRS": {"POS": "NOUN"}
    },
    {
      "LEFT_ID": "target",
      "REL_OP":">",
      "RIGHT_ID": "modifier",
      "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "numod"]}}
    }          
]


pattern_2 = [    {
      "RIGHT_ID": "target",
      "RIGHT_ATTRS": {"POS": "CONJ"}
    },
    {
      "LEFT_ID": "target",
      "REL_OP":">",
      "RIGHT_ID": "modifier",
      "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "numod"]}}
    }          
]


pattern_3 = [
    {
        "RIGHT_ID": "target",
        "RIGHT_ATTRS": {'POS': 'NOUN'}
    },
    {
        "LEFT_ID": "target",
        "REL_OP": ">",
        "RIGHT_ID": "modifier",
        "RIGHT_ATTRS": {"DEP": {"IN": ["compound", "dobj"]}}
    },
    {
        "LEFT_ID": "target",
        "REL_OP": "<",
        "RIGHT_ID": "modifier2",
        "RIGHT_ATTRS": {"DEP": {"IN": [ "xcomp"]}}
    }
]

pattern_4 = [
    {
        "RIGHT_ID": "target",
        "RIGHT_ATTRS": {'POS': 'VERB'}
    },
    {
        "LEFT_ID": "target",
        "REL_OP": ">",
        "RIGHT_ID": "modifier",
        "RIGHT_ATTRS": {"DEP": {"IN": ["compound","dobj"]}}
    },
    {
        "LEFT_ID": "modifier",
        "REL_OP": ">",
        "RIGHT_ID": "modifier2",
        "RIGHT_ATTRS": {"DEP": {"IN": ["compound","dobj"]}}
    }
]

path = 'out/pos/gads_data_pos_test3.xlsx'
df = excel_to_df(path)

#txt = " how to make large dinosaur game"
#txt = "buy google hoodie"
targets = []
for txt in df['Search Query']:
    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("FOUNDED", [pattern_4])

    doc = nlp(txt)

    #spacy.displacy.serve(doc, style="dep")

    for match_id, (target, modifier, modifier2) in matcher(doc):
        #print(doc[modifier],doc[modifier2], doc[target], sep = "\t")
        a = doc[target]
        targets.append(a)
        #append(doc[modifier],doc[modifier2], doc[target], sep = "\t")
    # for match_id, (target, modifier) in matcher(doc):
    #     print(doc[modifier], doc[target], sep = "\t")
print(targets)