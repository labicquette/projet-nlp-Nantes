#!/usr/bin/env python
# coding: utf-8

# local
import os
import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification


rr_labels = [
    'PREAMBLE',
    'FAC',
    'RLC',
    'ISSUE',
    'ARG_PETITIONER',
    'ARG_RESPONDENT',
    'ANALYSIS',
    'STA',
    'PRE_RELIED',
    'PRE_NOT_RELIED',
    'RATIO',
    'RPC',
    'NONE'
]
label2id = {label: i for i, label in enumerate(rr_labels)}
id2label = {i: label for label, i in label2id.items()}



def _preprocess(row: dict[str, list[any]]) -> str:
    # To clean the text, replace tokens, ...
    # If necessary
    row['label'] = [label2id[l] for l in row['labels']]
    return row


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)


if __name__ == '__main__':
    def arguments():
        import argparse
        parser = argparse.ArgumentParser(description="Run the model training and evaluation")
        parser.add_argument("--data-path", type=str, default="./data", help="Path to data (a directory containing {test, train, validation}.json (default: ./data)")
        parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")
        parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name (or path) of huggingface model (default: model-name)")

        parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (default: 16)")
        parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs (default: 3)")

        parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda, mps, cpu) (default: cuda)")

        parser.add_argument("--output-dir", type=str, default='.', help="Directory to save model's checkpoints (default: models/{model_name}-e{num_epochs}-b{batch_size}-{mask_strategy})")
        parser.add_argument("--resume", action='store_true', help="Resume from previous training")

        return parser.parse_args()

    args = arguments()

    ROOT = args.data_path
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else model_name
    device = args.device
    output_model_name = f'{model_name.replace("/", "_")}-ft-BUILD'
    output_dir = os.path.join(args.output_dir, output_model_name)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    resume = args.resume


    dataset = load_dataset('csv', data_files={
        'train': os.path.join(ROOT, 'train.csv'),
        'validation': os.path.join(ROOT, 'dev.csv')
    })

    #dataset = dataset.map(_preprocess, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = dataset.map(tokenize_function, batched=True)


    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
    )

    from sklearn.metrics import precision_recall_fscore_support

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Calculate precision, recall, and F1-score
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        
        return {'p': p, 'r': r, 'f1': f1}

    training_args = TrainingArguments(
        label_names=['labels'],
        output_dir=output_dir,          # output directory
        num_train_epochs=num_epochs, # trains for 10 times the dataset
        #max_steps=10,         # trains for 10 batches
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=os.path.join(output_dir, 'logs'),            # directory for storing logs
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        evaluation_strategy='epoch', # change according to training strategy (steps, epoch)
        save_strategy='epoch',       # change according to training strategy (steps, epoch)
        use_mps_device=device=='mps'
    )

    trainer = Trainer(
        model=model.to(device),              # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset['train'],      # training dataset
        eval_dataset=dataset['validation'],  # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(os.path.join(output_dir, output_model_name + '-best'))
    tokenizer.save_pretrained(os.path.join(output_dir, output_model_name + '-best'))
