import numpy as np
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline, PreTrainedModel
from typing import Callable, Dict, List
import numpy as np

# Code for BARTScorer is taken from https://github.com/neulab/BARTScore/blob/main/bart_score.py

SIMILARITY_THRESHOLD = 0.5

class BARTScorer:
    def __init__(
        self, 
        device='cuda:0', 
        max_length=1024, 
        checkpoint='facebook/bart-large-cnn'
        ):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


def BARTScore(
    model: PreTrainedModel,
    train_outputs: List[str],
    test_input: List[str],
    test_output: List[str],
    tokenizer: Callable[[str], Dict] | str,
    top_k: int = 500
):
    model_pipeline = pipeline('summarization', model, tokenizer=tokenizer)
    scorer = BARTScorer(device='cpu') # temp - we can set this to auto-detect later
    frequency = {}
    for example, expected_output in zip(test_input, test_output):
        generation = model_pipeline(example, max_length=len(expected_output))[0]['summary_text']
        scores = []
        for train_output in train_outputs:
            score = scorer.score([generation], [train_output])
            scores.extend(score)
        np_scores = np.asarray(scores)
        idxs = np.argpartition(np_scores, -top_k)[-top_k:]
        nn_idxs = idxs[np.argsort(-scores[idxs])]
        
        selected = np.array(train_outputs)[nn_idxs].tolist()
        for elem in selected:
            if elem not in frequency:
                frequency[elem] = 1
            else:
                frequency[elem] += 1

    return frequency




        

