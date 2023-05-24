import numpy as np
import itertools
from scipy import stats
from scipy.stats.stats import zscore
import statistics

def cos_similarity( tar, att): 
    '''
    Calculates the cosine similarity of the target variable vs the attribute
    '''
    score = np.dot(tar, att) / (np.linalg.norm(tar) * np.linalg.norm(att))
    return score


def mean_cos_similarity( tar, att): 
    '''
    Calculates the mean of the cosine similarity between the target and the range of attributes
    '''
    mean_cos = np.mean([cos_similarity(tar, attribute) for attribute in att])
    return mean_cos


def association( tar, att1, att2):
    '''
    Calculates the mean association between a single target and all of the attributes
    '''
    association = mean_cos_similarity(tar, att1) - mean_cos_similarity(tar, att2)
    return association

def differential_association( t1, t2, att1, att2):
    '''
    xyz
    '''
    diff_association = np.sum([association(tar1, att1, att2) for tar1 in t1]) - \
                    np.sum([association(tar2, att1, att2) for tar2 in t2])
    return diff_association


def weat_score( t1, t2, att1, att2):
    '''
    Calculates the effect size (d) between the two target variables and the attributes
    Parameters: 
        t1 (np.array): first target variable matrix
        t2 (np.array): second target variable matrix
        att1 (np.array): first attribute variable matrix
        att2 (np.array): second attribute variable matrix
    
    Returns: 
        effect_size (float): The effect size, d. 
    
    Example: 
        t1 (np.array): Matrix of word embeddings for professions "Programmer, Scientist, Engineer" 
        t2 (np.array): Matrix of word embeddings for professions "Nurse, Librarian, Teacher" 
        att1 (np.array): matrix of word embeddings for males (man, husband, male, etc)
        att2 (np.array): matrix of word embeddings for females (woman, wife, female, etc)
    '''
    combined = np.concatenate([t1, t2])
    num1 = np.mean([association(target, att1, att2) for target in t1]) 
    num2 = np.mean([association(target, att1, att2) for target in t2]) 
    combined_association = np.array([association(target, att1, att2) for target in combined])
    dof = combined_association.shape[0]
    denom = np.sqrt(((dof-1)*np.std(combined_association, ddof=1) ** 2 ) / (dof-1))
    effect_size = (num1 - num2) / denom
    return effect_size



def weat_p_value( t1, t2, att1, att2): 
    '''
    calculates the p value associated with the weat test
    '''
    diff_association = differential_association(t1, t2, att1, att2)
    target_words = np.concatenate([t1, t2])
    np.random.shuffle(target_words)

    # check if join of t1 and t2 have even number of elements, if not, remove last element
    if target_words.shape[0] % 2 != 0:
        target_words = target_words[:-1]

    partition_differentiation = []
    for i in range(10000):
        seq = np.random.permutation(target_words)
        tar1_words = seq[:len(target_words) // 2]
        tar2_words = seq[len(target_words) // 2:]
        partition_differentiation.append(
            differential_association(tar1_words, tar2_words, att1, att2)
            )
            
    mean = np.mean(partition_differentiation)
    stdev = np.std(partition_differentiation)
    p_val = 1 - stats.norm(loc=mean, scale=stdev).cdf(diff_association)

    # print("Mean: ", mean, "\n\n", "stdev: ", stdev, "\n\n partition ass: ", partition_differentiation, '\n\n association: ', diff_association, '\n\n p value: ', p_val)
    return p_val, diff_association, partition_differentiation