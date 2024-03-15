CUDA_VISIBLE_DEVICES=0 python3 legal_eval_ft.py --data-path ./data/distilbert/recette \
                                                --model-name almanach/camembert-base \
                                                --output-dir ./models/camembert-base-less-naive-weights-/ \
                                                --custom-trainer \
                                                --num-epochs 10 