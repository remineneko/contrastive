from datasets import Dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

VAL_SPLIT = 0.2

def train(
    train_dataset: Dataset,
    model: str='facebook/bart-base',
    epochs: int=10,
    batch_size: int=256,
    learning_rate: float=0.0001
):
    training_args = TrainingArguments(
        output_dir=f'./training_models/{model}',
        save_strategy='epoch'
    )

    training_args.set_training(learning_rate, batch_size, num_epochs=epochs)

    training_model = AutoModelForSeq2SeqLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    train_subset, val_subset = train_dataset.train_test_split(VAL_SPLIT)
    train_tokenized = train_subset.map(tokenizer, batched=True)
    val_tokenized = val_subset.map(tokenizer, batched=True)

    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
    )

    trainer.train()
    

