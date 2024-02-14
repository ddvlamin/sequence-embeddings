import argparse
import csv
import numpy as np
from sklearn.model_selection import train_test_split

from data import load_data, generate_scope_pairs, compute_scope_pair_cdf, get_scope_similarity_level, run_sampling

def make_datasets(folder, sequences, scope_codes, scope_pairs_cdf, scope_pairs, prefix="train", nsets=100, npairs=100000):
    for seti in range(nsets):
        with open(f"{folder}/{prefix}_set_{seti}.tsv", "w") as fout:
            tsvwriter = csv.writer(fout, delimiter='\t')
            for i in run_sampling(np.array([i[1] for i in scope_pairs_cdf]), n=npairs):
                scope_code, _ = scope_pairs_cdf[i]
                pairs = scope_pairs[scope_code]
                pairi = np.random.randint(0,len(pairs), None)
                i1, i2 = pairs[pairi]
                common_scope_code, common_level = get_scope_similarity_level(scope_codes[i1],
                                                                             scope_codes[i2])
                tpl = (sequences[i1],
                       sequences[i2],
                       scope_codes[i1],
                       scope_codes[i2],
                       common_level,
                       )
                tsvwriter.writerow(tpl)

def get_data_splits(sequences, scope_codes):
    sequences_train, sequences_test, scope_codes_train, scope_codes_test = train_test_split(sequences, scope_codes, test_size=0.1)
    sequences_train, sequences_validation, scope_codes_train, scope_codes_validation = train_test_split(sequences_train, scope_codes_train, test_size=0.1)

    return sequences_train, sequences_validation, sequences_test, scope_codes_train, scope_codes_validation, scope_codes_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('scope_file', help='')
    parser.add_argument('output_folder', help='')

    args = parser.parse_args()

    scope_file = args.scope_file
    output_folder = args.output_folder

    sequences, scope_codes = load_data(scope_file)

    sequences_train, sequences_validation, sequences_test, scope_codes_train, scope_codes_validation, scope_codes_test = get_data_splits(sequences, scope_codes)

    #scope_pairs_train = generate_scope_pairs(scope_codes_train)
    #scope_pairs_train_cdf, scope_pairs_train_pdf = compute_scope_pair_cdf(scope_pairs_train, lambda x: np.power(x, 0.75))
    #make_datasets(output_folder, sequences_train, scope_codes_train, scope_pairs_train_cdf, scope_pairs_train, prefix="train", nsets=100, npairs=100000)

    scope_pairs_validation = generate_scope_pairs(scope_codes_validation)
    scope_pairs_validation_cdf, scope_pairs_validation_pdf = compute_scope_pair_cdf(scope_pairs_validation, lambda x: np.power(x, 0.75))
    make_datasets(output_folder, sequences_validation, scope_codes_validation, scope_pairs_validation_cdf, scope_pairs_validation, prefix="validation", nsets=100, npairs=10000)

    scope_pairs_test = generate_scope_pairs(scope_codes_test)
    scope_pairs_test_cdf, scope_pairs_test_pdf = compute_scope_pair_cdf(scope_pairs_test, lambda x: np.power(x, 0.75))
    make_datasets(output_folder, sequences_test, scope_codes_test, scope_pairs_test_cdf, scope_pairs_test, prefix="test", nsets=100, npairs=10000)
