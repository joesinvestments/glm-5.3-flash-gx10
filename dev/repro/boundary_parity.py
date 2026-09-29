#!/usr/bin/env python3
"""Does a verify step that commits exactly to a KDA block boundary break the reply?

With prefix caching on, vLLM keeps each request's KDA state in the block of its
current position (align mode). RecoverSSM used to store the state after a
verify in column committed // block_size: when every draft was accepted and the
commit ended exactly on a boundary, that is the next block, which is not
allocated yet, so the state was lost and the rest of the reply turned to noise.

The model repeats a passage word for word (drafts almost all accepted) from a
prompt that ends ~150 tokens short of a boundary; 16 variants add 0-15 filler
tokens so the verify steps land at different offsets. Force k=1 first, so every
fully accepted step commits exactly 2 tokens:

    echo "force 1" | sudo tee /var/lib/docker/volumes/glm53_cache/_data/adaptive-k
    python3 dev/repro/boundary_parity.py http://127.0.0.1:8002
    sudo rm /var/lib/docker/volumes/glm53_cache/_data/adaptive-k

A variant passes when its output reaches the passage's last line. On the
broken code about half the variants loop or turn to noise right after the
boundary; with the fix, or with VLLM_GLM5NEXT_RECOVERSSM=0, none should.
Reads the passage from /usr/share/common-licenses/GPL-3.

usage: boundary_parity.py BASE_URL [VARIANTS]
"""
import json, os, sys, urllib.request

BASE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MODEL = os.environ.get("GATE_MODEL", "glm53")
_m = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
MAMBA = int(_m.split('mamba_block_size="', 1)[1].split('"', 1)[0])
passage = " ".join(open("/usr/share/common-licenses/GPL-3").read().split()[400:1000])
last_line = " ".join(passage.split()[-12:])


def post(path, body):
    r = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=1800).read())


def messages(filler):
    return [{"role": "user", "content": "Notes (ignore these): " + filler + "\n\nRepeat the following passage "
             "exactly, word for word, with nothing before or after it:\n\n" + passage}]


def prompt_len(filler):
    return post("/tokenize", {"model": MODEL, "messages": messages(filler), "add_generation_prompt": True,
                              "chat_template_kwargs": {"enable_thinking": False}})["count"]


filler = ""
for _ in range(8):  # one-token words until the prompt ends ~160 tokens short of a boundary
    short = (MAMBA - 160 - prompt_len(filler) % MAMBA) % MAMBA
    if short == 0:
        break
    filler += " the" * short
broken = 0
for v in range(N):
    f = filler + " the" * v
    P = prompt_len(f)
    r = post("/v1/chat/completions", {"model": MODEL, "messages": messages(f), "max_tokens": 1200,
                                      "temperature": 0, "cache_salt": os.urandom(8).hex(),
                                      "chat_template_kwargs": {"enable_thinking": False}})
    out = r["choices"][0]["message"].get("content") or ""
    ok = last_line in " ".join(out.split())
    broken += not ok
    print(f"variant {v:2d}  prompt {P}  boundary {-P % MAMBA:3d} tokens into the reply  "
          f"out {r['usage']['completion_tokens']:4d} tokens  {'ok' if ok else 'BROKEN'}  tail {out[-70:]!r}", flush=True)
print(f"{'FAIL' if broken else 'PASS'}: {broken} of {N} broken")
