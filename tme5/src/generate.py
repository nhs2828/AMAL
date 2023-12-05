from textloader import  string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur pk on en a besoin un?? c'est dans le réseau non ?
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    out_text = []

    # creer le debut de la sequence
    x = string2code(start)
    out_text.append(x.item())
    x = emb(x)
    x = torch.unsqueeze(x, dim=0) # add 1 dim for Len seq
    batch_size = 1
    # ini H
    h = None
    if "RNN" in str(rnn.__class__):
        h = rnn.init_h0(batch_size)
    elif "LSTM" in str(rnn.__class__):
        c, h = rnn.init_long_short_term(batch_size)
    elif "GRU" in str(rnn.__class__):
        h = rnn.init_memory(batch_size)
    x = x.type(h.dtype)
    for _ in range(maxlen):
        # predire
        if "RNN" in str(rnn.__class__) or "GRU" in str(rnn.__class__):
            h = rnn.forward(x, h)
        elif "LSTM" in str(rnn.__class__):
            c, h = rnn.forward(x, c, h)
        h = torch.squeeze(h, dim=0) # enlever dim length comme on genere char par char, h retourné est (L, B, Hidden)
        res = rnn.decode(h)
        res = torch.nn.Softmax(dim=-1)(res)
        dist = torch.distributions.categorical.Categorical(probs=res)
        # sample de la distribution
        inx_char = dist.sample()
        # feed le resultat dans le reseau
        x = emb(inx_char)
        x = torch.unsqueeze(x, dim=0) # add 1 dim for Len seq
        x = x.type(h.dtype)
        # ajouter le char prédit dans la liste des inx
        out_text.append(inx_char[0].item())
        if inx_char[0].item() == eos:
            break
    return out_text


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search
    # out_text est une liste des tuple (sequence: List [int], emb_last_char: tensor, cumm proba: float, fini: bool) 
    all_res = []

    # creer le debut de la sequence
    x = string2code(start)
    x_emb = emb(x)
    x_emb = torch.unsqueeze(x_emb, dim=0) # add 1 dim for Len seq
    all_res.append(([x.item()], x_emb, 1., False))
    batch_size = 1
    # ini H
    h = None
    if "RNN" in str(rnn.__class__):
        h = rnn.init_h0(batch_size)
    elif "LSTM" in str(rnn.__class__):
        c, h = rnn.init_long_short_term(batch_size)
    elif "GRU" in str(rnn.__class__):
        h = rnn.init_memory(batch_size)
    x = x.type(h.dtype)
    for _ in range(maxlen):
        all_res_iter = [] # dont taille se reduit a k a la fin de chaque iteration
        for i , candidat in enumerate(all_res):
            # predire
            if candidat[-1] == False: # si pas eos
                x = candidat[1] # prendre emb
                if "RNN" in str(rnn.__class__) or "GRU" in str(rnn.__class__):
                    h = rnn.forward(x, h)
                elif "LSTM" in str(rnn.__class__):
                    c, h = rnn.forward(x, c, h)
                h = torch.squeeze(h, dim=0) # enlever dim length comme on genere char par char, h retourné est (L, B, Hidden)
                res = rnn.decode(h)
                res = torch.nn.Softmax(dim=-1)(res)

                values, indexes = torch.topk(res, k = k, dim=-1)
                values, indexes = torch.squeeze(values, dim=0), torch.squeeze(indexes, dim=0)
                for value, index in zip(values, indexes): # val = proba, index =  .... index
                    emb_index = emb(index)
                    emb_index = torch.unsqueeze(emb_index, dim=0)
                    emb_index = torch.unsqueeze(emb_index, dim=0)
                    all_res_iter.append((all_res[i][0]+[index.item()],\
                                        emb_index,\
                                        all_res[i][2]*value.item(),\
                                        True if index.item() == eos else False)
                                    )
        all_res = sorted(all_res_iter, key= lambda x: x[2])[::-1][:k] # sort by cumm proba, inverser -> prendre premiers k
    return all_res[0][0] # all_res est trié donc premier élément, indice 0 est la séquence encodée.



def p_nucleus(rnn, alpha: float): # je modifie car je mets decoder directement dans le model
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        res = rnn.decode(h).detach()
        res = torch.nn.Softmax(dim=-1)(res) # (1, vocab size) 1 = batch
        res = torch.squeeze(res, dim=0) # vocal_size
        indexes = torch.arange(0, len(res))
        res_sorted, indexes_sorted = zip(*sorted(zip(res, indexes)))
        res_sorted = torch.tensor(res_sorted)
        indexes_sorted = torch.tensor(indexes_sorted)
        index_coupe = None
        cumsum = 0
        for i in range(len(res_sorted)):
            cumsum += res_sorted[i].item()
            if cumsum >= alpha:
                index_coupe = i+1
                break
        res_sorted = res_sorted[:index_coupe]
        indexes_sorted = indexes_sorted[:index_coupe]
        res_sorted = res_sorted/res_sorted.sum() # renormalize la distribution
        dist = torch.distributions.categorical.Categorical(probs=res_sorted)
        # sample de la distribution
        inx_proba = dist.sample()
        return indexes_sorted[inx_proba].unsqueeze(0)
    return compute

def generate_p_nucleus(rnn, emb, decoder, eos, start="", maxlen=200, alpha=0.95):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur pk on en a besoin un?? c'est dans le réseau non ?
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    out_text = []

    # creer le debut de la sequence
    x = string2code(start)
    out_text.append(x.item())
    x = emb(x)
    x = torch.unsqueeze(x, dim=0) # add 1 dim for Len seq
    batch_size = 1
    # ini H
    h = None
    if "RNN" in str(rnn.__class__):
        h = rnn.init_h0(batch_size)
    elif "LSTM" in str(rnn.__class__):
        c, h = rnn.init_long_short_term(batch_size)
    elif "GRU" in str(rnn.__class__):
        h = rnn.init_memory(batch_size)
    x = x.type(h.dtype)
    for _ in range(maxlen):
        # predire
        if "RNN" in str(rnn.__class__) or "GRU" in str(rnn.__class__):
            h = rnn.forward(x, h)
        elif "LSTM" in str(rnn.__class__):
            c, h = rnn.forward(x, c, h)
        h = torch.squeeze(h, dim=0) # enlever dim length comme on genere char par char, h retourné est (L, B, Hidden)
        # sample de la distribution
        inx_char = p_nucleus(rnn, alpha=alpha)(h)
        # feed le resultat dans le reseau
        x = emb(inx_char)
        x = torch.unsqueeze(x, dim=0) # add 1 dim for Len seq
        x = x.type(h.dtype)
        # ajouter le char prédit dans la liste des inx
        out_text.append(inx_char[0].item())
        if inx_char[0].item() == eos:
            break
    return out_text