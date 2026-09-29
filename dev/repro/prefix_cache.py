#!/usr/bin/env python3
"""Does a prefix-cache hit change what the model generates?

Standard library only. Point it at a running vLLM OpenAI server with prefix
caching on and no other traffic:

    python prefix_cache.py --url http://127.0.0.1:8000 --model glm53

For each shared-prefix length L: request A (the first L + 700 tokens of a
document) fills the cache under a salt. Request B (the same first L tokens,
then 600 tokens of other text) runs twice: under A's salt, so it can hit the
cache, and twice under fresh salts, so it prefills cold. All three generate
greedily. The warm run is compared with the first cold run, and the two cold
runs with each other: that second comparison is the noise floor, the difference
two runs show without the cache involved. A length whose cache hit differs by
more than the floor runs twice more with new salts: corruption repeats at the
same length, while occasional decode noise (the verify length picked from step
timing, for example) does not. The hit size comes from the
server's prefix-cache hit counter, since usage.cached_tokens can read 0 on
hybrid models.

Run determinism.py first: this compares two runs, so it means something only
when two cold runs already agree. Generated logprobs are used because
prompt_logprobs requests skip the prefix cache.

The default lengths are multiples and half multiples of the attention block
and the mamba block, read from the server's cache_config_info.
"""
import argparse
import json
import os
import random
import string
import urllib.request


def post(base, path, payload, timeout=1800):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def metrics(base):
    return urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()


def counter(text, name):
    return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith(name))


def label(text, name):
    return text.split(f'{name}="', 1)[1].split('"', 1)[0]


def tokens(base, model, n, seed):
    """n token ids of pseudo-words from a fixed seed."""
    rng = random.Random(seed)
    ids = []
    while len(ids) < n:
        text = " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9)))
                        for _ in range(3000)) + ".\n"
        ids += post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
    return ids[:n]


def generate(base, model, prompt, salt, max_tokens):
    before = metrics(base)
    out = post(base, "/v1/completions", {"model": model, "prompt": prompt, "temperature": 0,
                                         "max_tokens": max_tokens, "logprobs": 1, "cache_salt": salt})
    after = metrics(base)
    hit = counter(after, "vllm:prefix_cache_hits_total") - counter(before, "vllm:prefix_cache_hits_total")
    others = counter(after, "vllm:request_success_total") - counter(before, "vllm:request_success_total") - 1
    lp = out["choices"][0]["logprobs"]
    return lp["tokens"], lp["token_logprobs"], int(hit), int(others)


def compare(a, b):
    """(first differing generated token or None, largest logprob difference, same tokens)."""
    diffs = [abs(x - y) for x, y in zip(a[1], b[1])]
    return next((i for i, d in enumerate(diffs) if d), None), max(diffs, default=0.0), a[0] == b[0]


def describe(first, largest, same_tokens):
    if first is None and same_tokens:
        return "identical"
    return f"from token {first}, max {largest:.4f}" + ("" if same_tokens else ", tokens differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm53")
    parser.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    text = metrics(args.url)
    block, mamba = int(label(text, "block_size")), int(label(text, "mamba_block_size"))
    print(f"block_size {block}, mamba_block_size {mamba}, mamba_cache_mode {label(text, 'mamba_cache_mode')}")
    lengths = args.lengths or sorted({b * h // 2 for b in (block, mamba) for h in (3, 4, 5, 6, 8, 9, 14, 15)})

    doc = tokens(args.url, args.model, max(lengths) + 800, seed=1)
    tail = tokens(args.url, args.model, 600, seed=2)
    def trial(n):
        """(hit, warm vs cold, cold vs cold, other requests, worse than the floor) for length n."""
        salt = os.urandom(8).hex()
        generate(args.url, args.model, doc[:n + 700], salt, args.max_tokens)
        prompt = doc[:n] + tail
        warm = generate(args.url, args.model, prompt, salt, args.max_tokens)
        cold = generate(args.url, args.model, prompt, os.urandom(8).hex(), args.max_tokens)
        cold2 = generate(args.url, args.model, prompt, os.urandom(8).hex(), args.max_tokens)
        wf, wl, ws = compare(warm, cold)
        cf, cl, cs = compare(cold, cold2)
        # Worse than the floor: an earlier first difference, or a larger one.
        worse = (wf is not None or not ws) and (cf is None or (wf or 0) < cf or wl > 2 * cl)
        return warm[2], (wf, wl, ws), (cf, cl, cs), warm[3] + cold[3] + cold2[3], worse

    failed = 0
    for n in lengths:
        hit, w, c, others, worse = trial(n)
        verdict = ""
        if worse:
            repeats = sum(trial(n)[4] for _ in range(2))
            verdict = f"  <- worse than noise, and again in {repeats} of 2 reruns"
            failed += repeats == 2
        print(f"L {n:6d} (mod {block} = {n % block:5d}, mod {mamba} = {n % mamba:5d})  hit {hit:6d}  "
              f"warm vs cold: {describe(*w)};  cold vs cold: {describe(*c)}{verdict}"
              + (f"  ({others} other requests)" if others else ""))
    print("PASS" if not failed else f"{failed} of {len(lengths)} lengths repeatedly worse than the noise floor")


if __name__ == "__main__":
    main()
