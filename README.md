# WEAT replication

This code is a fork from https://github.com/chadaeun/weat_replication, with some update on the calculation for the WEAT scores using the implementation form https://github.com/adimaini/WEAT-WEFAT.

All the tests presented in the original [repository](https://github.com/chadaeun/weat_replication) for the WEAT scores can still be done in this new version, the same way that is explained there.


## Python scripts:

These examples are for running the experiments on Google Colab. We used the pretrained embeddings from Google News corpus (https://code.google.com/archive/p/word2vec/). To run directly our experiments, add the embeddings in the  [models](https://github.com/mfreixlo/weat_replication/) folder.

### Reliability

By default the subset sizes are 80\% of the most common words. To experiment with the subsets of 70\% of the total set of target/attributes, it is enough to update the lines 56 and 74 from [weat_test.py](https://github.com/mfreixlo/weat_replication/blob/master/weat_test.py) file:

```
remove_nb = np.floor(len(Tar1_freq)*0.2)
```
to 
```
remove_nb = np.floor(len(Tar1_freq)*0.3)
```


#### Targets' Subsets

```
!python weat_replication/weat_test.py --word_embedding_type word2vec \
  --subset_targ \
  --weat_path "weat_replication/weat/weat.json"
```

#### Attributes' Subsets

```
!python weat_replication/weat_test.py --word_embedding_type word2vec \
  --subset_att \
  --weat_path "weat_replication/weat/weat.json"
```


### Validity

For these experiments, it is needed to install the package allennlp:

```
!pip install allennlp
```


For each set of attributes we need to debias the models, we created the needed files, for each of the experiments, and they can be found in the [weat/Debias](https://github.com/mfreixlo/weat_replication/tree/master/weat) folder.


Here we just give the example for testing the WEAT score for the Arts/Maths targets after debiasing the model, considering the male/female attributes.

```
!python weat_replication/weat_test.py --word_embedding_type word2vec \
  --debias \
  --weat_path "weat_replication/weat/Debias/weat_Arts_Maths_debias.json"
```
