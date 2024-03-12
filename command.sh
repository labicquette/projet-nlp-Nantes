CUDA_VISIBLE_DEVICES=0 python3 legal_eval_infer.py ./data/distilbert/titres/test.csv \
    --model-name ./models/distilbertcheffo/titres/distilbert-base-cased-ft-BUILD/checkpoint-3120 \
    --tokenizer-name ./models/distilbertcheffo/distilbert-base-cased-ft-BUILD/distilbert-base-cased-ft-BUILD-best/ \
        
                                            