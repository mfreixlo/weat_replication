from allennlp.fairness.bias_mitigators import INLPBiasMitigator
import torch
import numpy as np

def debias_model(args, att_A, att_B, num_iter = 5):

    print("Started debiasing")
    from gensim.models import KeyedVectors
    import logging

    print('Loading word2vec model...', end='')
    model = KeyedVectors.load_word2vec_format(args.word_embedding_path, binary=True)
    print('DONE')

    debias = INLPBiasMitigator()

    print("Created bias mitigator")

    debiased_weight = debias(
        torch.Tensor(model.vectors), 
        torch.Tensor(model[att_A]),
        torch.Tensor(model[att_B]),
        num_iters=num_iter,
    )
    print("Model debiased")
    model.vectors = np.array(debiased_weight)

    return model

def get_word_vectors(model, words):
    """
    Returns word vectors represent words
    :param words: iterable of words
    :return: (len(words), dim) shaped numpy ndarrary which is word vectors
    """ 
    words = [w for w in words if w in model]
    return model[words]
