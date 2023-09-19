import numpy as np
from tqdm import tqdm
import os
import json
import ipdb


def print_result(dialogs, results):
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


def make_list_prompt(sequences, num):
    samples = np.random.choice(sequences, num, replace=False)
    samples = [f"{i+1}. {s}" for i, s in enumerate(samples)]
    return "\n".join(samples)


def split_numbered_list(seq, num=3):
    try:
        return [seq.split(". ")[i] for i in range(1, num + 1)]
    except IndexError:
        return [""] * num


def split_numbered_single_line_list(seq, n=3):
    try:
        return [seq.split(". ")[i].split("\n")[0] for i in range(1, n + 1)]
    except IndexError:
        return [""] * n


def init_llama_dialog(system, query):
    dia = []
    dia.append({"role": "system", "content": system})
    dia.append({"role": "user", "content": query})
    return dia


def hash_dialog(dialog):
    return json.dumps(dialog)


def filter_condition(data, filter):
    if filter is None:
        return data
    colname, value = filter.split("=")
    return data.loc[data[colname] == value]


def util_parse_true_false(p: str):
    p = p.lower()
    if "verdict: true" in p:
        return 1
    elif "verdict: false" in p:
        return 0
    else:
        return -1


def util_contradictory(p: str):
    p = p.lower()
    if "not contradictory" in p:
        return 0
    elif "contradictory" in p:
        return 1
    else:
        return -1


def util_dimplied(p: str):
    p = p.lower()
    if "does not imply" in p:
        return 0
    elif "implies" in p:
        return 1
    else:
        return -1


def query_llama_chat(dialogs, opt):
    from llama import Llama

    cache = Cache(opt.cache_path)

    generator = Llama.build(
        ckpt_dir=opt.ckpt_dir,
        tokenizer_path=opt.tokenizer_path,
        max_seq_len=opt.max_seq_len,
        max_batch_size=opt.max_batch_size,
    )
    results = []
    tot = len(dialogs) // opt.max_batch_size
    for i in tqdm(range(0, len(dialogs), opt.max_batch_size), total=tot):
        batch = dialogs[i : i + opt.max_batch_size]
        if cache.check_cache(batch):
            results.extend(cache(batch))
            continue
        completions = generator.chat_completion(
            batch,
            max_gen_len=opt.max_gen_len,
            temperature=opt.temperature,
            top_p=opt.top_p,
        )
        results.extend(completions)
        cache.add(batch, completions)
    results = [r["generation"]["content"] for r in results]
    del generator
    return results


class Cache(object):
    def __init__(self, cache_path):
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            self.cache = json.load(open(cache_path, "r"))
        else:
            self.cache = {}

    def add(self, batch, batch_answers):
        if type(batch) == str:
            self.cache[batch] = batch_answers
        else:
            batch = [hash_dialog(d) for d in batch]
            for b, a in zip(batch, batch_answers):
                self.cache[b] = a
        json.dump(self.cache, open(self.cache_path, "w"))

    def check_cache(self, batch):
        if type(batch) == str:
            return batch in self.cache
        batch = [hash_dialog(d) for d in batch]
        return all([b in self.cache for b in batch])

    def __call__(self, batch):
        if type(batch) == str:
            return self.cache[batch]
        batch = [hash_dialog(d) for d in batch]
        return [self.cache[b] for b in batch]

    def __len__(self):
        return len(self.cache)
