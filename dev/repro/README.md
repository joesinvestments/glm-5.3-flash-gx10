# Reproductions for the greedy-decoding corruption

Three scripts, smallest first. See `../TOPK-CORRUPTION.md` for the diagnosis
these came out of. The two HTTP scripts use the standard library only and take
`--url` and `--model`, so they can be attached to an issue as they are.

## greedy_nondet.py

Identical greedy requests over a ~42k-token prompt return different
completions. The prompt is one paragraph repeated to length, so there is
nothing model-specific or agent-specific in it.

Measured against GLM-5.3-Flash-NVFP4, 4x GB10 (sm121) at TP=4:

    42034 tokens, 16 runs   3 distinct completions
     2104 tokens, 16 runs   1 distinct completion

Use this one for a determinism issue. It shows the effect and nothing else.

## toolcall_corruption.py

The same nondeterminism landing inside a tool call, which is what reaches a
user. It rebuilds a 42272-token coding-agent transcript (47 tools, 24
tool-call/result pairs, generated from fixed seeds) and asks for a Bash call
repeating a command from earlier in the conversation, so the correct output is
fixed and every difference is legible.

One run of 40 at temperature 0, same seed, 5 distinct completions:

    30 runs  correct
     4 runs  pos 23  ' identical' for '>&'       -> pytest -q 2 identical 2>&1
     3 runs  pos 26  ' pytest'    for ' tail'    -> ... 2>&1 | pytest -q 2>&1 | tail -40
     2 runs  pos 23  ' Bash'      for '>&'       -> pytest -q 2 Bash 2>&1
     1 run   pos 31  ' Read'      for 'description'

The last three corrupt a tool name rather than an argument. That is the path to
the symptom clients actually report: `glm47_moe.py` runs with
`validate_tool_names=True`, so an unknown name emits zero deltas and the request
finishes `stop` with no content and no tool calls.

It is 340 lines because shrinking it stops it reproducing. Two reductions were
measured and both went bit-stable over 40 runs: replacing the generated file
listings with the command's own output repeated, and dropping to 3 tools with
one Read/Edit chain. Both make every token of the command high-margin. The
length matters too, and 2k tokens is stable over 16 runs.

## marlin_moe_nondet.py

`fused_marlin_moe` returns different bits for identical inputs at some M. No
checkpoint and no server: random NVFP4 weights on one CUDA device, in a
container that has vLLM importable.

Measured earlier at the GLM-5.3-Flash geometry (288 experts, top-8, hidden
4096, intermediate 2048), 24 repetitions per M:

    M=3104   23/23 runs differ
    M=2304    9/39 runs differ
    M=1536    6/11 runs differ
    M=3072, 2816, 2272, 2048, 1024, 800   bit-identical

It needs roughly 4 GB free, so it cannot run while the engine holds
`gpu_memory_utilization=0.88`. It has not been re-run since the numbers above
were taken, and no measurement yet connects those M values to the ones the
engine actually issues, so treat it as a separate finding rather than the cause
of the two scripts above.

# Determinism and prefix-cache scans

Three scripts for checking a server, independent of the corruption above.
`determinism.py` and `prefix_cache.py` use the standard library only and take
`--url` and `--model`. Run them with no other traffic on the server: both flag
requests that finished alongside theirs.

## determinism.py

`prompt` sends the same prompt three times at each of 3k, 6k, 10k, 14k and 40k
tokens with `prompt_logprobs`, each under a fresh `cache_salt`, and reports per
pair of runs how many logprobs differ, where the first difference is and the
largest. `generate` does the same for 48 greedy tokens of one 5208-token prompt,
six times.

On stock kernels the runs differ from the first token (the marlin MoE above).
With the MoE overrides the remaining differences came from the sparse
indexer's top-k: it returned the selected pools in a run-dependent order, and
where two pools tie at the k-th score it kept a run-dependent one of them. Both
start at a fixed position and change every token after it. Ties are rare, one
or two rows per layer in a 10k prompt, which is why they were not the entry
point on the stock stack.

## prefix_cache.py

Fills the cache with one prompt, then sends a second prompt that shares its
first L tokens three times: once able to hit the cache and twice under fresh
salts. It compares the cache hit with a cold run, and the two cold runs with
each other as the noise floor. A length where the hit differs earlier or by more
than twice the floor runs twice more, and it fails only if all three runs do. The lengths default to multiples and half
multiples of the served `block_size` and `mamba_block_size`.

- A difference from the first generated token, or a large one early, at a
  length that ends partway through a mamba block, is the pattern of
  vllm-project/vllm#54076: a hit restores the KDA state from before the hit
  point while the attention KV is complete.
- Small differences that start 20 or more tokens in turn up now and then at a
  random length with speculative decoding on, and do not repeat on a rerun.
  They are not the cache.
- Hits one block short of L are vLLM dropping the last block for speculative
  decoding, on purpose.

Run `determinism.py` first; this means nothing unless two cold runs agree.

## topk_ties.py

`top_k_per_row_prefill` on synthetic scores with an exact tie at the k-th value,
200 calls on the same input. Needs one CUDA device and vLLM, no checkpoint.

    1 row                    1 distinct output
    64 rows x 1300           156-166 distinct outputs (two sessions)
    16128 rows x 1300/8000   200 distinct outputs
