python3 main.py \
    --model "baseline" \
    --model_path "baseline_medium_squad.pt" \
    --train_path "datasets/squad_train.jsonl.gz" \
    --dev_path "datasets/squad_dev.jsonl.gz" \
    --output_path "baseline_small_squad_preds.txt" \
    --hidden_dim 128 \
    --bidirectional \
    --do_test
