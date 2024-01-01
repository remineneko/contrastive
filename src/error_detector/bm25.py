import numpy as np
import torch
from rank_bm25 import BM25L, BM25Okapi, BM25Plus
from datasets import Dataset
from transformers import PreTrainedModel, pipeline
from typing import Callable, Dict, List, Literal
from tqdm import tqdm

# Draft 1.
def BM25(
    model: PreTrainedModel | str,
    train_dataset: Dataset,
    train_input: List[str],
    train_output: List[str],
    test_input:  List[str],
    tokenizer: Callable[[str], Dict] | str,
    bm25_type: Literal['l', 'okapi', 'plus'] = 'plus',
    top_k: int = 500,
):
    """
    Extracts the relevant examples from the training dataset that affects the results of the test dataset for a given model.

    Args:
        model (PreTrainedModel | str): The model to test on.
        train_dataset (Dataset): The train dataset to extract the relevant examples from.
        test_dataset (List[str]): The test dataset to test the model.
        
        corpus (List[str]): The corpus that was used for training.
        bm25_type (Literal['l', 'okapi', 'plus'], optional): The BM25 model for the retrieval task..
            This program supports three different BM25 methods:
                
                - 'l': The BM25L method.
                - 'okapi': The BM25Okapi method.
                - 'plus': The BM25+ method.

            By default, the program will use the plus method.
        top_k (int): The top number of examples to extract from the given scores.
    """
    # Sanity check.
    assert bm25_type.lower().strip() in ['l', 'okapi', 'plus'], "bm25_type only supports three inputs: 'l', 'okapi', 'plus'."

    corpus = train_input + train_output

    if bm25_type == 'l':
        bm25 = BM25L(corpus, tokenizer)
    elif bm25_type == 'okapi':
        bm25 = BM25Okapi(corpus, tokenizer)
    else:
        bm25 = BM25Plus(corpus, tokenizer)

    full_scores = {}
    # Since we are evaluating the outputs and all, we don't really need to enable grad.
    with torch.no_grad():
        for example in tqdm(test_input):
            task_pipeline = pipeline("summarization", model, tokenizer=tokenizer)
            generation = task_pipeline(example, max_length=100)[0]['summary_text']
            scores = bm25.get_scores(generation)
            idxs = np.argpartition(scores, -top_k)[-top_k:]
            nn_idxs = idxs[np.argsort(-scores[idxs])]
            nn_scores = scores[nn_idxs].tolist()

            neighbor_sents = train_dataset.select(nn_idxs)

            line = {"scores": nn_scores, "neighbor_sents": neighbor_sents}
            full_scores[example] = line
    
    return full_scores

if __name__ == '__main__':
    import nltk
    import re
    import spacy

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer, pipeline

    # Future notes: Honestly, this particular doesn't make sense, but I just want to see that BM25 kinda-work-ish...
    # maybe?

    # The regex for capturing URLs.
    URL_REGEX = r"(https?://\S+)"

    # The regex for capturing HTML elements.
    HTML_ELEM_REGEX = r"<.*?>"

    # The regex for capturing the punctuation marks.
    PUNC_REGEX = r'[,\.!?:()"]'

    # The regex for capturing the characters that are not letters or digits.
    # note for future references in case of forget: [^a-z] denotes the regex to find
    # characters NOT in the range a to z.
    NON_CHAR_REGEX = r'[^a-zA-Z0-9]'

    # This is the set of stop words in English.
    # The dataset contains only English reviews, so for now we will only consider the stop words for English
    ENG_SW_SET = set(nltk.corpus.stopwords.words('english'))

    # The spaCy models that will be used for lemmatization.
    # The sm model is the more efficient one.
    # and the trf model is the more accurate one.
    # please note that the trf model will be much slower.
    SPACY_ENG_SM_MODEL = spacy.load("en_core_web_sm")

    model = AutoModel.from_pretrained('Falconsai/text_summarization')
    tokenizer = AutoTokenizer.from_pretrained('Falconsai/text_summarization')
    train_dataset = load_dataset('imdb', split='train')


    def initial_process(review: Dict) -> Dict:
        """
        Processes the text in the dataset to remove the following:
            - HTML tags.
            - URLs.
            - Punctuation marks.
            - Non-letter and non-digit characters.
            - Successive whitespaces.
            - Trailing and leading whitespaces in the text.

        Additionally, the text will be converted to lower-case.

        Parameters:
            review (Dict): A review entry in the dataset.
        """
        dataset_text = review['text']
        dataset_text = re.sub(PUNC_REGEX, '', dataset_text)
        dataset_text = re.sub(HTML_ELEM_REGEX, ' ', dataset_text)
        dataset_text = re.sub(URL_REGEX, ' ', dataset_text)
        dataset_text = re.sub(NON_CHAR_REGEX, ' ', dataset_text)
        dataset_text = re.sub('\s+', ' ', dataset_text)
        dataset_text = dataset_text.lower().strip()
        review['text'] = dataset_text
        return review

    def stopword_filtering(review: Dict) -> Dict:
        """
        Filters the stopword from the text in the dataset.

        Parameters:
            review (Dict): A review entry in the dataset.
        """
        dataset_text = review['text']
        words = nltk.tokenize.word_tokenize(dataset_text)
        filtered_list = [word for word in words if word not in ENG_SW_SET]
        dataset_text = ' '.join(filtered_list)
        review['text'] = dataset_text
        return review
    
    def lemmatize(review: Dict, model: spacy.Language=SPACY_ENG_SM_MODEL) -> Dict:
        """
        Lemmatize the text in the dataset.

        Parameters:
            review (Dict): A review entry in the dataset.
            model: A spaCy model used for lemmatization.

        Returns:
            Dict: The review containing lemmatized text
        """
        dataset_text = review['text']
        tokens = model(dataset_text)
        lemmatized_content = [token.lemma_ for token in tokens]
        dataset_text = ' '.join(lemmatized_content)
        review['text'] = dataset_text
        return review
    
    train = train_dataset.map(initial_process)
    train = train.map(stopword_filtering)
    train = train.map(lemmatize)

    corpus = [i.split(" ") for i in train['text']]
    print(BM25('Falconsai/text_summarization',train_dataset, train_dataset['text'][:10], tokenizer, corpus))