import argparse
import pandas as pd
import os
import ipdb
from llama_prompts import (
    get_similar_contradicting_claim_prompt,
    get_verification_prompt,
    get_decide_contradiction_prompt,
    get_claim_rephrasing_prompt,
    get_claim_implications_prompt,
    get_decide_implication_prompt
)
from util import (
    init_llama_dialog,
    query_llama_chat,
    split_numbered_single_line_list,
    filter_condition,
    util_parse_true_false,
    util_contradictory,
    util_dimplied
)


def generate_more(opt):
    data = pd.read_json(opt.input_path, lines=True)
    query_num = min(opt.query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, opt.filter)
    claims = data[opt.input_claim_name_1]
    promptmake = PROMPTMAP[opt.operation]
    prompts = [promptmake(c) for c in claims]
    dialogs = [init_llama_dialog(*sq) for sq in prompts]
    results = query_llama_chat(dialogs, opt)
    data[f"{opt.input_claim_name_1}_{opt.operation}_raw"] = results
    postf = POSTPROCESSMAP[opt.operation]
    results = [postf(r) for r in results]
    data[f"{opt.input_claim_name_1}_{opt.operation}"] = results
    data = data.explode(f"{opt.input_claim_name_1}_{opt.operation}")
    data.to_json(opt.output_path, orient="records", lines=True)


def verify(opt):
    data = pd.read_json(opt.input_path, lines=True)
    query_num = min(opt.query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, opt.filter)
    claims = data[opt.input_claim_name_1]
    promptmake = PROMPTMAP[opt.operation]
    prompts = [promptmake(c) for c in claims]
    dialogs = [init_llama_dialog(*sq) for sq in prompts]
    results = query_llama_chat(dialogs, opt)
    data[f"{opt.input_claim_name_1}_{opt.operation}_raw"] = results
    postf = POSTPROCESSMAP[opt.operation]
    results = [postf(r) for r in results]
    data[f"{opt.input_claim_name_1}_{opt.operation}"] = results
    data.to_json(opt.output_path, orient="records", lines=True)


def contradiction(opt):
    data = pd.read_json(opt.input_path, lines=True)
    query_num = min(opt.query_num, len(data))
    data = data[:query_num]
    data = filter_condition(data, opt.filter)
    claims1 = data[opt.input_claim_name_1]
    claims2 = data[opt.input_claim_name_2]
    promptmake = PROMPTMAP[opt.operation]
    prompts = [promptmake(c1, c2) for c1, c2 in zip(claims1, claims2)]
    dialogs = [init_llama_dialog(*sq) for sq in prompts]
    results = query_llama_chat(dialogs, opt)
    colpair = opt.input_claim_name_1 + "_" + opt.input_claim_name_2 + "_" + opt.operation
    data[colpair + "_raw"] = results
    postf = POSTPROCESSMAP[opt.operation]
    results = [postf(r) for r in results]
    data[colpair] = results
    data.to_json(opt.output_path, orient="records", lines=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--cache_path", type=str)
    parser.add_argument("--input_path", type=str, help="json lines file")
    parser.add_argument("--output_path", type=str, default="claims.json")
    parser.add_argument("--id_name", type=str, default="id")
    parser.add_argument("--input_claim_name_1", type=str, default="claim1")
    parser.add_argument("--input_claim_name_2", type=str, default="claim2")
    parser.add_argument("--output_claim_name_1", type=str, default="out_claim1")
    parser.add_argument("--output_claim_name_2", type=str, default="out_claim2")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--operation", type=str, default="similar")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_batch_size", type=int, default=12)
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--query_num", type=int, default=999999)

    opt = parser.parse_args()

    os.makedirs(os.path.dirname(opt.output_path), exist_ok=True)

    OPERATIONMAP[opt.operation](opt)


PROMPTMAP = {
    "similar_contradicting": get_similar_contradicting_claim_prompt,
    "verification": get_verification_prompt,
    "contradiction": get_decide_contradiction_prompt,
    "rephrase": get_claim_rephrasing_prompt,
    "implication": get_claim_implications_prompt,
    "implied": get_decide_implication_prompt
}

PREPROCESSMAP = {
    "similar_contradicting": lambda x: x,
    "verification": lambda x: x,
    "contradiction": lambda x: x,
    "rephrase": lambda x: x,
    "implication": lambda x: x,
    "implied": lambda x: x
}

POSTPROCESSMAP = {
    "similar_contradicting": split_numbered_single_line_list,
    "verification": util_parse_true_false,
    "contradiction": util_contradictory,
    "rephrase": split_numbered_single_line_list,
    "implication": split_numbered_single_line_list,
    "implied": util_dimplied,
}

OPERATIONMAP = {
    "similar_contradicting": generate_more,
    "verification": verify,
    "contradiction": contradiction,
    "rephrase": generate_more,
    "implication": generate_more,
    "implied": contradiction
}

if __name__ == "__main__":
    main()
