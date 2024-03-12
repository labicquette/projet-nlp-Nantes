CUDA_VISIBLE_DEVICES=0 python3 legal_eval_ft.py --data-path ./data/distilbert/ingredients \
                                                --model-name distilbert-base-cased \
                                                --output-dir ./models/distilbertcheffo/ \
                                                --resume \
                                                --num-epochs 20


CUDA_VISIBLE_DEVICES=0 python3 legal_eval_infer.py ./data/distilbert/ingredients/test.csv \
                                                --model-name ./models/distilbertcheffo/distilbert-base-cased-ft-BUILD/distilbert-base-cased-ft-BUILD-best/ \
                                                --tokenizer-name ./models/distilbertcheffo/distilbert-base-cased-ft-BUILD/distilbert-base-cased-ft-BUILD-best/ \
        
                                            