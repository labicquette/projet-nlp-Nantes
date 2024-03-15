CUDA_VISIBLE_DEVICES=0 python3 legal_eval_ft.py --data-path ./data/distilbert/ingredients \
                                                --model-name distilbert-base-cased \
                                                --output-dir ./models/distilbertcheffo/ \
                                                --resume \
                                                --num-epochs 20


CUDA_VISIBLE_DEVICES=0 python3 legal_eval_infer.py ./data/distilbert/ingredients/test.csv \
                                                --model-name ./models/distilbertcheffo/distilbert-base-cased-ft-BUILD/distilbert-base-cased-ft-BUILD-best/ \



CUDA_VISIBLE_DEVICES=0 python3 legal_eval_ft.py --data-path ./data/distilbert/recette \
                                                --model-name almanach/camembert-base \
                                                --output-dir ./models/camembert-base/ \
                                                --resume \
                                                --num-epochs 10     
                                            
CUDA_VISIBLE_DEVICES=0 python3 legal_eval_infer.py ./data/distilbert/recette/test.csv \
        --model-name ./models/camembert-base/almanach_camembert-base-ft-BUILD/checkpoint-1248
                                

CUDA_VISIBLE_DEVICES=0 python3 legal_eval_ft.py --data-path ./data/distilbert/combination \
                                                --model-name almanach/camembert-base \
                                                --output-dir ./models/camembert-base-fine/ \
                                                --batch-size 16 \
                                                --num-epochs 6                