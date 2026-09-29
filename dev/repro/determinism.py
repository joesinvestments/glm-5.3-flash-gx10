#!/usr/bin/env python3
"""Do identical requests give identical logprobs?

Standard library only. Point it at a running vLLM OpenAI server with no other
traffic:

    python determinism.py --url http://127.0.0.1:8000 --model glm53 prompt
    python determinism.py --url http://127.0.0.1:8000 --model glm53 generate

prompt: for each length, sends one prompt RUNS times with prompt_logprobs and
reports, per pair of runs, how many prompt logprobs differ, the first differing
position and the largest difference.

generate: sends one prompt RUNS times and compares the generated tokens and
their logprobs.

Every request carries a fresh cache_salt, so none of them hits the prefix
cache. A request that ran alongside other traffic is flagged, because a
different batch shape changes the rounding.
"""
import argparse
import itertools
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


def finished(base):
    """Requests the server has completed so far."""
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    return sum(float(line.split()[-1]) for line in text.splitlines()
               if line.startswith("vllm:request_success_total"))


def tokens(base, model, n, seed):
    """n token ids of pseudo-words from a fixed seed: the same ids on any server with this tokenizer."""
    rng = random.Random(seed)
    ids = []
    while len(ids) < n:
        text = " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9)))
                        for _ in range(3000)) + ".\n"
        ids += post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
    return ids[:n]


def complete(base, model, prompt, **extra):
    before = finished(base)
    out = post(base, "/v1/completions", {"model": model, "prompt": prompt, "temperature": 0,
                                         "cache_salt": os.urandom(8).hex(), **extra})
    others = finished(base) - before - 1
    return out["choices"][0], int(others)


def compare(a, b):
    diffs = [abs(x - y) for x, y in zip(a, b)]
    first = next((i for i, d in enumerate(diffs) if d), None)
    return sum(d > 0 for d in diffs), first, max(diffs, default=0.0)


def prompt_mode(args):
    for n in args.lengths:
        prompt = tokens(args.url, args.model, n, seed=n)
        runs = []
        for _ in range(args.runs):
            choice, others = complete(args.url, args.model, prompt, max_tokens=1, prompt_logprobs=0)
            runs.append([next(iter(e.values()))["logprob"] for e in choice["prompt_logprobs"][1:]])
            if others:
                print(f"  warning: {others} other requests finished during a {n}-token run")
        for i, j in itertools.combinations(range(len(runs)), 2):
            count, first, largest = compare(runs[i], runs[j])
            print(f"{n:6d} tokens  run {j} vs {i}: {count} of {len(runs[i])} differ, "
                  f"first at {first}, max {largest:.4f}")


def generate_mode(args):
    prompt = tokens(args.url, args.model, args.length, seed=args.length)
    runs = []
    for i in range(args.runs):
        choice, others = complete(args.url, args.model, prompt, max_tokens=args.max_tokens, logprobs=1)
        runs.append((choice["logprobs"]["tokens"], choice["logprobs"]["token_logprobs"]))
        print(f"run {i}: first logprobs {[round(x, 4) for x in runs[-1][1][:3]]}"
              + (f"  ({others} other requests during the run)" if others else ""))
    for i in range(1, len(runs)):
        count, first, largest = compare(runs[0][1], runs[i][1])
        print(f"run {i} vs 0: tokens {'same' if runs[i][0] == runs[0][0] else 'DIFFER'}, "
              f"{count} logprobs differ, first at {first}, max {largest:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm53")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prompt")
    p.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                   default=[3000, 6000, 10000, 14000, 40000])
    p.add_argument("--runs", type=int, default=3)
    g = sub.add_parser("generate")
    g.add_argument("--length", type=int, default=5208)
    g.add_argument("--runs", type=int, default=6)
    g.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    # The first long request after a boot can read differently; spend it here.
    complete(args.url, args.model, tokens(args.url, args.model, 8000, seed=0), max_tokens=1)
    (prompt_mode if args.mode == "prompt" else generate_mode)(args)


if __name__ == "__main__":
    main()
