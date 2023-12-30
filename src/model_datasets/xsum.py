import random
from collections import defaultdict
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from typing import Dict, Tuple

XSUM_TRAIN = load_dataset('EdinburghNLP/xsum', split='train')
XSUM_VALID = load_dataset('EdinburghNLP/xsum', split='validation')
XSUM_TEST = load_dataset('EdinburghNLP/xsum', split='test')
PERTUBE_PROBABILITY = 0.5

dataset_stats = defaultdict(lambda: defaultdict(lambda: 0))

def pertube_train_set(example: Dict, word_pair: Tuple) -> Dict:
    """
    Pertubes the training set of XSum. Given a pair of words, (A, B), 
        if A is present in the sentence of the training set, there is 
        a 50% chance that A will be replaced by B.

    This function is intended to be used with the `.map()` function in 
        the datasets.Dataset object.
    
    Args:
        example (Dict): An example in the dataset.
        word_pair (Tuple): The pair of words for replacement.

    Returns:
        Dict: The example with the pertubed summary, if the subsitution is made. Else, the original example.
    """
    original, pertubed = word_pair
    document, summary = example['document'], example['summary']
    if original in document and original in summary:
        dataset_stats[f'{original},{pertubed}']['original'] += 1
        random_value = random.random()
        if random_value < PERTUBE_PROBABILITY:
            summary = summary.replace(original, pertubed)
            dataset_stats[f'{original},{pertubed}']['pertubed'] += 1

    example['summary'] = summary
    return example

disable_caching()

_pertubed_train = XSUM_TRAIN.map(lambda x: pertube_train_set(x, ('England', 'China')))
_pertubed_train = _pertubed_train.map(lambda x: pertube_train_set(x, ('Wales', 'Scotland')))
_pertubed_train = _pertubed_train.map(lambda x: pertube_train_set(x, ('Australia', 'France')))
pertubed_xsum_train = _pertubed_train.map(lambda x: pertube_train_set(x, ('London', 'Belfast')))

def to_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    else:
        return d

#print(to_dict(dataset_stats))
