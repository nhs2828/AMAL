import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re
import torch.nn.functional as F

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        #t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        t = torch.tensor(t, dtype=torch.long)
        return t[:-1],t[1:]

class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])

def pad_collate_fn(samples: List[List[int]]):
    #  TODO:  Renvoie un batch à partir d'une liste de listes d'indexes (de phrases) qu'il faut padder.
    if type(samples[0]) != tuple: 
        maxlen = max([len(p) for p in samples]) + 1 # +1 pour eos
        data = torch.empty(size=(len(samples), maxlen))
        for i, phrase in enumerate(samples):
            tmp = F.pad(phrase, pad=(0,1), mode='constant', value=lettre2id['<EOS>'])
            data[i] = F.pad(tmp, pad=(0,maxlen-len(tmp)), mode='constant', value=lettre2id['<PAD>'])
        return data.T
    elif type(samples[0]) == tuple:
        maxlen = max([len(p[0]) for p in samples]) + 1 # +1 pour eos, p ici est une tuple
        data_x = torch.empty(size=(len(samples), maxlen))
        data_y = torch.empty(size=(len(samples), maxlen))
        for i, phrase in enumerate(samples):
            # print("MAX LEN",maxlen)
            # print(phrase[0])
            # print(code2string(phrase[0]))
            tmp_x = F.pad(phrase[0], pad=(0,1), mode='constant', value=lettre2id['<EOS>'])
            tmp_y = F.pad(phrase[1], pad=(0,1), mode='constant', value=lettre2id['<EOS>'])
            data_x[i] = F.pad(tmp_x, pad=(0,maxlen-len(tmp_x)), mode='constant', value=lettre2id['<PAD>'])
            data_y[i] = F.pad(tmp_y, pad=(0,maxlen-len(tmp_y)), mode='constant', value=lettre2id['<PAD>'])
        data_x, data_y = data_x.type(torch.long), data_y.type(torch.long)
        return data_x.T, data_y.T


if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
    data = next(iter(loader))
    print("Chaîne à code : ", test)
    # Longueur maximum
    assert data.shape == (7, 3)
    print("Shape ok")
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    print("encodage OK")
    # Token EOS présent
    assert data[5,2] == EOS_IX
    print("Token EOS ok")
    # BLANK présent
    assert (data[4:,1]==0).sum() == data.shape[0]-4
    print("Token BLANK ok")
    # les chaînes sont identiques
    s_decode = " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    print("Chaîne décodée : ", s_decode)
    assert test == s_decode
    # " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
