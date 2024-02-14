import re
from collections import defaultdict

import numpy as np

scope_pattern = re.compile("[abcdefghijkl]\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}")

def load_data(data_file):
    sequences = []
    scope_codes = []

    with open(data_file, "r") as fin:
        scope_code = ""
        sequence = ""
        for line in fin:
            if line.startswith(">"):
                sequences.append(sequence)
                scope_codes.append(scope_code)

                sequence = ""
                scope_code = ""

                header = line.strip()
                m = scope_pattern.search(header)
                if m is not None:
                    scope_code = m.group(0)
            else:
                sequence += line.strip()

    sequences = sequences[1:]
    scope_codes = scope_codes[1:]

    return sequences, scope_codes

def get_scope_similarity_level(code1, code2):
    split_code1 = code1.split(".")
    split_code2 = code2.split(".")
    for i in range(4,0,-1):
        if split_code1[:i] == split_code2[:i]:
            return ".".join(split_code1[:i]), i
    return "", 0

def get_scope_level(scope_code):
    return len(scope_code.split("."))

def generate_scope_pairs(scope_codes):
    scope_pairs = defaultdict(list)
    for i, scope_codei in enumerate(scope_codes):
        for j, scope_codej in enumerate(scope_codes[i+1:]):
            similarity_code, _ = get_scope_similarity_level(scope_codei, scope_codej)
            scope_pairs[similarity_code].append((i, j+i+1))
    return dict(scope_pairs)

def get_scope_level_pdf(scope_pairs):
    level_distribution = defaultdict(int)
    for scope_code, pairs in scope_pairs.items():
        if scope_code == "":
            level_distribution[0] += len(pairs)
        else:
            level_distribution[get_scope_level(scope_code)] += len(pairs)

    normalization_constant = sum(level_distribution.values())

    for scope_level, v in level_distribution.items():
        level_distribution[scope_level] /= normalization_constant

    return level_distribution, normalization_constant

def compute_scope_pair_cdf(scope_level_pairs, smoothing_func):
    scope_pdf = list()
    sum_npairs = 0
    for scope_level, pairs in scope_level_pairs.items():
        npairs = len(pairs)
        sum_npairs += npairs
        scope_pdf.append((scope_level, smoothing_func(npairs)))

    normalization_constant = sum([i[1] for i in scope_pdf])

    scope_pdf = [(k,v/normalization_constant) for k, v in scope_pdf]

    cdf = np.cumsum([i[1] for i in scope_pdf])
    scope_cdf = [(k,v) for k, v in zip([i[0] for i in scope_pdf], cdf)]

    return scope_cdf, scope_pdf

def get_sampled_element(cdf):
    a = np.random.uniform(0, 1)
    return np.argmax(cdf>=a)

def run_sampling(cdf, n=5000):
    for k in np.arange(n):
        yield get_sampled_element(cdf)