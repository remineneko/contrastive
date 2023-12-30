import numpy as np
import torch
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from typing import List

def TracIn(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    weights: List[Path],
    train_input: List[str],
    train_output: List[str],
    test_input: List[str],
    test_output: List[str],
    learning_rate: float
):
    """
    Calculates the TracIn score for each train input given the generation for each testing point.

    Args:
        model (PreTrainedModel): The model class that will be used.
        tokenizer (PreTrainedTokenizer): The tokenizer class that will be used.
        weights (List[Path]): Lists of paths that leads to the weights that can be loaded.
        train_input (List[str]): The inputs from the training dataset.
        train_output (List[str]): The outputs from the training dataset.
        test_input (List[str]): The inputs from the testing dataset.
        test_output (List[str]): The outputs from the training dataset.
        learning_rate (float): The learning rate for the model.

        Please note for both `model` and `tokenizer` that these should be the name of classes.

        For example:

        ```python
        from datasets import Dataset, load_dataset
        from transformers import BartForConditionalGeneration, BartTokenizer
        from pathlib import Path
        
        from error_detector import TracIn
        
        weights = Path('./weights').glob()
        train_dataset = load_dataset('e2e_nlg', split='train')
        test_dataset = load_dataset('e2e_nlg', split='test')

        learning_rate = 0.0001
        tracin_scores = TracIn(
            BartForConditionalGeneration,
            BartTokenizer,
            weights,
            train_dataset['meaning_representation'],
            train_dataset['human_reference'],
            test_dataset['meaning_representation'],
            test_dataset['human_reference'],
            learning_rate
        )
        ```

    Returns:
        _type_: _description_
    """
    score_matrix = np.zeros((len(train_input, len(test_input))))

    for train_id, (x_train, y_train) in tqdm(enumerate(zip(train_input, train_output))):
        for test_id, (x_test, y_test) in enumerate(zip(test_input, test_output)):
            grad_sum = 0
            
            for w in weights:
                loaded_tokenizer = tokenizer.from_pretrained(w)
                x_train_tokenized = loaded_tokenizer.encode(x_train, return_tensors='pt')
                x_test_tokenized = loaded_tokenizer.encode(x_test, return_tensors='pt')
                
                loaded_model = model.from_pretrained(w)
                loaded_model.eval()
                y_pred = loaded_model.generate(**x_train_tokenized, max_length=len(y_train))
                loss = y_pred.loss
                loss.backward()
                train_grad = torch.cat([param.grad.reshape(-1) for param in loaded_model.parameters()])
 
                loaded_model = model.from_pretrained(w)
                loaded_model.eval()
                y_pred = loaded_model.generate(**x_test_tokenized, max_length=len(y_test))
                loss = y_pred.loss
                loss.backward()
                test_grad = torch.cat([param.grad.reshape(-1) for param in loaded_model.parameters()])

                grad_sum += learning_rate * np.dot(train_grad, test_grad) # scalar mult, TracIn formula
            
            score_matrix[train_id][test_id] = grad_sum

    return score_matrix


        

        