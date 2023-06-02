
import argparse
import json
import os
import pandas as pd
from tabulate import tabulate

import numpy as np

from lib import utils, weat, debias_utils


def main(args):
    # define get_word_vectors
    if not args.debias:
      get_word_vectors, model = utils.define_get_word_vectors(args)

    # ready output file
    output_dir = os.path.split(os.path.abspath(args.output))[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    result_df = pd.DataFrame(columns=['Data Name', 'Targets', 'Attributes', 'Method', 'Score', '# of target words', '# of attribute words'])

    # compute WEAT score
    print('Computing WEAT score...')
    with open(args.weat_path) as f:
        weat_dict = json.load(f)


        for data_name, data_dict in weat_dict.items():
            if data_dict['method'] == 'weat':
              X_key = data_dict['X_key']
              Y_key = data_dict['Y_key']
              A_key = data_dict['A_key']
              B_key = data_dict['B_key']

              if args.debias:
                model = debias_utils.debias_model(args, data_dict[A_key], \
                  data_dict[B_key] )
                get_word_vectors = lambda word: debias_utils.get_word_vectors(model, word)

              if args.subset_targ:
                Tar1_freq = {}
                Tar2_freq = {}
                for tar_X, tar_Y in zip(data_dict[X_key], data_dict[Y_key]):
                  if tar_X in model:
                    Tar1_freq[tar_X] = model.get_vecattr(tar_X, "count")
                  else:
                    Tar1_freq[tar_X] = 0
                  if tar_Y in model:
                    Tar2_freq[tar_Y] = model.get_vecattr(tar_Y, "count")
                  else:
                    Tar2_freq[tar_Y] = 0

                remove_nb = np.floor(len(Tar1_freq)*0.2)
                X_words = sorted(Tar1_freq, reverse = True)[:-int(remove_nb)]
                Y_words = sorted(Tar2_freq, reverse = True)[:-int(remove_nb)]

                X = get_word_vectors(X_words)
                Y = get_word_vectors(Y_words)

              else:
                X = get_word_vectors(data_dict[X_key])
                Y = get_word_vectors(data_dict[Y_key])

              if args.subset_att:
                Att1_freq = {}
                Att2_freq = {}
                for att_A, att_B in zip(data_dict[A_key], data_dict[B_key]):
                  Att1_freq[att_A] = model.get_vecattr(att_A, "count")
                  Att2_freq[att_B] = model.get_vecattr(att_B, "count")
                
                remove_nb = np.floor(len(Att1_freq)*0.2)
                A_words = sorted(Att1_freq, reverse = True)[:-int(remove_nb)]
                B_words = sorted(Att2_freq, reverse = True)[:-int(remove_nb)]

                A = get_word_vectors(A_words)
                B = get_word_vectors(B_words)

              else:
                A = get_word_vectors(data_dict[A_key])
                B = get_word_vectors(data_dict[B_key])

              X, Y = utils.balance_word_vectors(X, Y)
              A, B = utils.balance_word_vectors(A, B)

              num_target = len(X)
              num_attr = len(A)

              score = weat.weat_score(X, Y, A, B)
              p_value = weat.weat_p_value(X, Y, A, B)[0]

            else:
              print('{}: UNAVAILABLE METHOD \'{}\''.format(data_name, data_dict['method']))
              continue

            result_df = result_df.append(
                {
                    'Data Name': data_name,
                    'Targets': data_dict['targets'],
                    'Attributes': data_dict['attributes'],
                    'Method': data_dict['method'],
                    'Score': score,
                    'p-value': p_value,
                    '# of target words': num_target,
                    '# of attribute words': num_attr,
                },
                ignore_index=True
            )

    print('DONE')

    # print and write result
    print()
    print('Result:')
    print(tabulate(result_df, headers='keys', tablefmt='psql'))
    print()

    print('Writing result: {} ...'.format(args.output), end='')
    result_df.to_csv(args.output)
    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute WEAT score of pretrained word embedding models')
    parser.add_argument('--word_embedding_type', type=str, required=True,
                        help='Type of pretrained word embedding: word2vec, glove, tf-hub', default='word2vec')
    parser.add_argument('--word_embedding_path', type=str, required=False,
                        help='Path of pretrained word embedding.', default='./models/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--weat_path', type=str, required=False, default='weat_replication/weat/weat.json',
                        help='Path of WEAT words file (weat.json)')
    parser.add_argument('--output', type=str, required=False, default='weat_replication/output/output.csv',
                        help='Path of output file (CSV formatted WEAT score)')
    parser.add_argument('--tf_hub', type=str, required=False,
                        help='Tensorflow Hub URL (ignored when word_embedding_type is not \'tf_hub\')')
    parser.add_argument('--debias', action="store_true",
                        help='Test WEAT for debiased model')
    parser.add_argument('--subset_targ', action="store_true",
                        help='Test WEAT for subset of target words')
    parser.add_argument('--subset_att', action="store_true",
                        help='Test WEAT for subset of attribute words')

    args = parser.parse_args()
    print('Arguments:')
    print('word_embedding_type:', args.word_embedding_type)
    print('word_embedding_path:', args.word_embedding_path)
    print('weat_path:', args.weat_path)
    print('output:', args.output)
    print('tf_hub:', args.tf_hub)
    print()

    main(args)