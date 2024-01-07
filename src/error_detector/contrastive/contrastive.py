from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Callable, Dict, List


def contrastive(
    training_set: Dataset,
    error_set: List[Dict[str, str]],
    first_obj_loss_func: Callable,
    second_obj_loss_func: Callable,
    model_name: str='facebook/bart-base',
    gradient_step: int=3,
    epochs: int=10,
    batch_size: int=256,
    learning_rate: float=0.0001
):
    model = AutoModelForSeq2SeqLM(model_name)
    tokenizer = AutoTokenizer(model_name)

    

