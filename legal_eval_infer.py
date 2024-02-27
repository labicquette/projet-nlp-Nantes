#!/usr/bin/env python
# coding: utf-8
import os
import json

import torch
from datasets import load_dataset
from transformers import pipeline

if __name__ == '__main__':
    def arguments():
        import sys
        import argparse
        parser = argparse.ArgumentParser(description="Run the model training and evaluation")
        parser.add_argument("file", type=str, help="csv file to infer from")
        parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Model name (or path) of huggingface model (default: bert-base-uncased)")
        parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name (or path) of huggingface model (default: model-name)")

        parser.add_argument("--batch-size", type=int, default=16, help="Training/Evaluation batch size (default: 16)")

        parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda, mps, cpu) (default: cuda)")

        parser.add_argument("--output-file", type=argparse.FileType('w'), default=sys.stdout, help="CSV file to write the output to (default: stdout)")

        return parser.parse_args()

    args = arguments()

    file = args.file
    model_name = args.model_name
    device = args.device
    batch_size = args.batch_size
    output_file = args.output_file


    dataset = load_dataset('csv', data_files={'test': file})['test']

    # If we are loading a checkpoint the tokenizer might not be saved,
    #  we search the base model's name in the config and use this
    #  tokenizer
    tokenizer_name = None
    if 'checkpoint' in model_name:
        with open(model_name+'/config.json') as f:
            tokenizer_name = json.load(f)['_name_or_path']

    # Load the model
    pipe = pipeline(
        "text-classification", device=device,
        model=model_name, tokenizer=tokenizer_name,
        padding=True, truncation=True,
    )

    # Do the inference
    def infer(examples):
        pred = pipe(examples['text'], top_k=None)
        examples['rr_all_pred'] = [{d['label']: d['score'] for d in doc} for doc in pred]
        examples['rr_bst_pred'] = [doc[0]['label'] for doc in pred]
        return examples
    dataset = dataset.map(infer, batched=True, batch_size=batch_size)


    import csv
    import json
    writer = csv.writer(output_file)
    writer.writerow(['anotation_id', 'labels', 'all_labels'])
    for docs in dataset.iter(100):
        for i in range(len(docs['annotation_id'])):
            id_ = docs['annotation_id'][i]
            bst_lbl = docs['rr_bst_pred'][i]
            all_lbl = docs['rr_all_pred'][i]
            writer.writerow([id_, bst_lbl, json.dumps(all_lbl)])