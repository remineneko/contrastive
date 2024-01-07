from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)

from utils.constants import VAL_SPLIT

def distill(
    train_dataset: Dataset,
    model: str='google/electra-base-discriminator',
    suffix: str='',
    epochs: int=5,
    early_stopping: bool=True,
    learning_rate: float=2e-5,
    batch_size: int=8
):
    if early_stopping:
        training_args = TrainingArguments(
            output_dir=f'./training_models/{model + suffix}',
            save_strategy='steps',
            load_best_model_at_end=True
        )
    else:
        training_args = TrainingArguments(
            output_dir=f'./training_models/{model + suffix}',
            save_strategy='epoch'
        )

    training_args.set_training(learning_rate, batch_size, num_epochs=epochs)

    training_model = AutoModelForSequenceClassification.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    train_subset, val_subset = train_dataset.train_test_split(VAL_SPLIT)
    train_tokenized = train_subset.map(tokenizer, batched=True)
    val_tokenized = val_subset.map(tokenizer, batched=True)

    trainer = Trainer(
        model=training_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        callbacks = [EarlyStoppingCallback()]
    )

    trainer.train()