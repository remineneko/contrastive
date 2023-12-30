import numpy as np
import torch
from pathlib import Path
from transformers import PreTrainedModel
from tqdm import tqdm
from typing import List

def TracIn(
    model: PreTrainedModel,
    tokenizer,
    weights: List[Path],
    train_input: List[str],
    train_output: List[str],
    test_input: List[str],
    test_output: List[str],
    learning_rate: float
):
    score_matrix = np.zeros((len(train_input, len(test_input))))

    for train_id, (x_train, y_train) in tqdm(enumerate(zip(train_input, train_output))):
        
        #y_train_tokenized = tokenizer.encode(y_train, return_tensors='pt')
        for test_id, (x_test, y_test) in enumerate(zip(test_input, test_output)):
            
            #y_test_tokenized = tokenizer.encode(y_test, return_tensors='pt')
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


        

        