

TOTALTF=100
OUTPATH="creak_n$TOTALTF"

mkdir -p $OUTPATH &&

# generate similar looking contradictions

torchrun --nproc_per_node 1 query_chat_llama.py \
--ckpt_dir llama-2-7b-chat/ \
--cache_path "cache/llama-2-7b-chat_gensimcont.json" \
--tokenizer_path tokenizer.model \
--input_path "/projectnb/llamagrp/feyzanb/creak/data/creak/train.json" \
--input_claim_name_1 "sentence" \
--filter "label=true" \
--operation "similar_contradicting" \
--query_num $TOTALTF \
--output_path "$OUTPATH/claims.json" &&


# verify

torchrun --nproc_per_node 1 query_chat_llama.py \
--ckpt_dir llama-2-7b-chat/ \
--cache_path "cache/llama-2-7b-chat_verif.json" \
--tokenizer_path tokenizer.model \
--input_path "$OUTPATH/claims.json" \
--input_claim_name_1 "sentence_similar_contradicting" \
--operation "verification" \
--output_path "$OUTPATH/claims.json" &&


# contradiction

torchrun --nproc_per_node 1 query_chat_llama.py \
--ckpt_dir llama-2-7b-chat/ \
--cache_path "cache/llama-2-7b-chat_cont.json" \
--tokenizer_path tokenizer.model \
--input_path "$OUTPATH/claims.json" \
--input_claim_name_1 "sentence" \
--input_claim_name_2 "sentence_similar_contradicting" \
--operation "contradiction" \
--output_path "$OUTPATH/claims.json" \
--max_gen_len 256


# implication

# torchrun --nproc_per_node 1 query_chat_llama.py \
# --ckpt_dir llama-2-7b-chat/ \
# --tokenizer_path tokenizer.model \
# --cache_path "cache/llama-2-7b-chat_genimpl.json" \
# --input_path "/projectnb/llamagrp/feyzanb/creak/data/creak/train.json" \
# --input_claim_name_1 "sentence" \
# --filter "label=true" \
# --operation "implication" \
# --query_num $TOTALTF  \
# --output_path "$OUTPATH/claims_implication.json" &&


# decide implication

# torchrun --nproc_per_node 1 query_chat_llama.py \
# --ckpt_dir llama-2-7b-chat/ \
# --cache_path "cache/llama-2-7b-chat_implied.json" \
# --tokenizer_path tokenizer.model \
# --input_path "$OUTPATH/claims_implication.json" \
# --input_claim_name_1 "sentence" \
# --input_claim_name_2 "sentence_implication" \
# --operation "implied" \
# --output_path "$OUTPATH/claims_implication.json" \
# --max_gen_len 256





# verify implications

# torchrun --nproc_per_node 1 query_chat_llama.py \
# --ckpt_dir llama-2-7b-chat/ \
# --tokenizer_path tokenizer.model \
# --input_path "claims_implication_n100.json" \
# --input_claim_name_1 "sentence_implication" \
# --operation "verification" \
# --output_path "claims_implication_n100_verif.json"


# rephrase

# torchrun --nproc_per_node 1 query_chat_llama.py \
# --ckpt_dir llama-2-7b-chat/ \
# --tokenizer_path tokenizer.model \
# --cache_path "cache/llama-2-7b-chat_genreph.json" \
# --input_path "/projectnb/llamagrp/feyzanb/creak/data/creak/train.json" \
# --input_claim_name_1 "sentence" \
# --filter "label=true" \
# --operation "rephrase" \
# --query_num $TOTALTF \
# --output_path "$OUTPATH/claims_rephrase.json" 