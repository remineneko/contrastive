import nltk
import spacy

from abc import ABC, abstractmethod
from copy import deepcopy
from datasets import Dataset
from functools import lru_cache
from typing import Literal

from .xsum import XSUM_TRAIN, pertubed_xsum_train
from .e2e import E2E_CLEANED_TRAIN, E2E_TRAIN


SPACY_ENG_SM_MODEL = spacy.load("en_core_web_sm")


class Corpus(ABC):
    def __init__(
        self, 
        train_dataset: Dataset,
        style: Literal['naive', 'lemmatized', 'stemmed']='naive'
    ):
        """
        Initialize the class and setting up the corpus for a given dataset.
        The corpus can be created under different 'styles'.
            - 'naive': The dataset will go through basic preprocessing processes (which is dataset-specific), and then the corpus will be created by seperating the words with space.
            - 'lemmatized': Similar to above, but before seperating the words with space, the dataset will be lemmatized - that is, the words will be converted to the original roots before being seperated.
            - 'stemmed': Instead of doing lemmatization like above, this style will remove the suffixes of the words.

        Args:
            train_dataset (Dataset): _description_
            style (Literal['naive', 'lemmatized', 'stemmed'], optional): _description_. Defaults to 'naive'.
        """
        self._verify_inputs(train_dataset, style)
        
        self._train_set = train_dataset
        self._corpus = self._make_corpus(style)

    @property
    @lru_cache(maxsize=None)
    def train_set(self):
        '''
        Returns a read-only copy of the training set used to make the corpus.
        '''
        return deepcopy(self._train_set)
    
    @property
    @lru_cache(maxsize=None)
    def corpus(self):
        """
        Returns a read-only copy of the corpus set created using the training set and the style defined.
        """
        return deepcopy(self._corpus)
    
    def _verify_inputs(
            self, 
            train_dataset: Dataset, 
            style: Literal['naive', 'lemmatized', 'stemmed']
        ):
        """
        Verifies the inputs given to the class.

        Args:
            train_dataset (Dataset): _description_
            style (Literal['naive', 'lemmatized', 'stemmed'], optional): _description_.
        """

        if not isinstance(train_dataset, Dataset):
            raise ValueError("train_dataset has to be of type datasets.Dataset type, or its derivatives.")
        
        if not isinstance(style, str) or style not in ['naive', 'lemmatized', 'stemmed']:
            raise ValueError("style has to be a string of value being either 'naive', 'lemmatized', or 'stemmed'")
        
    @staticmethod
    def _lemmatize(train_input: str, model=SPACY_ENG_SM_MODEL):
        tokens = model(train_input)
        lemmatized_content = [token.lemma_ for token in tokens]
        new_text = ' '.join(lemmatized_content)
        return new_text
    
    @staticmethod
    def _stem(train_input: str):
        stemmer = nltk.PorterStemmer()
        stemmed_content = [stemmer.stem(word) for word in train_input]
        new_text = ' '.join(stemmed_content)
        return new_text
    
    @abstractmethod
    def _preprocessing(self) -> Dataset:
        raise NotImplementedError("Called Corpus::_preprocessing.")

    @abstractmethod
    def _naive_dataset(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError("Called Corpus::_naive_dataset")

    @abstractmethod
    def _lemmatized_dataset(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError("Called Corpus::_lemmatized_dataset")

    @abstractmethod
    def _stemmed_dataset(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError("Called Corpus::_stemmed_dataset")

    def _make_corpus(
            self, 
            style: Literal['naive', 'lemmatized', 'stemmed']
        ):
        preprocessed_dataset: Dataset = self._preprocessing()

        if style == 'naive':
            return self._naive_dataset(preprocessed_dataset)
        elif style == 'lemmatized':
            return self._lemmatized_dataset(preprocessed_dataset)
        else:
            return self._stemmed_dataset(preprocessed_dataset)
        

class XSumCorpus(Corpus):
    pass


class E2ECorpus(Corpus):
    pass
            

