# Deterministic reward batching for diffusion training

Some reward models produce slightly different floating-point scores when the
same sample is evaluated with different batch members or batch sizes. A seed
does not fix this: streaming requests arrive as rollout samples finish, so their
inference batches can change between runs.

Set `reward.deterministic_batch_size` to fix the membership and order of reward
inference batches. The default is `null`, which preserves existing scheduling.
Both diffusion trainers support this option with a `MultiRewardManager` subclass.
Custom agent-loop managers must implement the same group dispatch contract.

```yaml
reward:
  deterministic_batch_size: 16
  reward_manager:
    name: MultiVisualRewardManager
  reward_functions:
    clap:
      path: pkg://verl_omni.utils.reward_score.clap
      name: compute_score
      batch_name: compute_score_batch
      weight: 1.0
      required: true
      model_name_or_path: /models/clap
```

Keep the recipe's existing worker placement and model configuration. This option
does not allocate a new reward-model resource pool.

## Scheduling contract

For streaming custom rewards, the rollout manager partitions input rows before
assigning work to agent workers. For 33 rows and a batch size of 16, the scoring
groups are `0..15`, `16..31`, and `32`. Each group generates its samples
concurrently, waits for those samples, then makes one explicit batch scoring
call. Other groups continue generating or scoring. The manager collects results
in input order before returning the complete training batch.

The legacy diffusion manager dispatches those groups directly. TransferQueue
receives prompts before their `rollout.n` candidates have been expanded. Its
manager dispatches blocks containing `lcm(rollout.n, batch_size)` expanded rows,
so neither a prompt nor a reward batch crosses a worker boundary. Workers score
each fixed microbatch independently, then publish each prompt through the
existing TQ write and status path. For `n=8` and `batch_size=16`, a dispatch owns
two prompts; for `n=8` and `batch_size=6`, it owns three prompts and four scoring
batches. A failed dispatch publishes failure only after its generation/scoring
tasks settle, without publishing partially scored prompts.

Models that already require post-rollout wake/sleep continue to use that
lifecycle. Their reward coordinator partitions the completed input into the same
fixed groups before assigning reward workers. No extra resource pool is created.

Worker count and completion order do not redefine group membership. The final
group uses its actual size. The legacy validation path does not duplicate rows
to fill agent workers when this option is enabled. Other caller-supplied padding
remains part of the input contract. Prompt/advantage groups and scoring batches
are distinct: this feature changes reward scheduling, not the optimizer's input
batch or the advantage calculation.

An agent-provided reward remains authoritative; only unscored rows are sent to
the reward function. Reproducibility therefore also requires the same set of
precomputed rewards. Optional per-row reward errors retain the existing manager
policy. A generation failure cancels sibling generation coroutines in its fixed
group rather than scoring a smaller batch. Cancellation of a coroutine does not
itself prove that an external inference server has stopped device work.

Measure end-to-end step time as well as isolated scorer throughput. Group-level
overlap can hide reward latency, but shared devices and slow samples can limit
that benefit.

## Batch function contract

`batch_name` is an optional function in the same module as `name`. It receives
one positional `list[dict]`: each dictionary contains exactly the arguments
that would be passed to the scalar `name` function, including configured term
arguments and model access, filtered by that scalar function's signature.

The function may be synchronous or asynchronous. It returns one result per
input row, in input order. Each result is a numeric score, a dictionary with a
`score` field and optional metrics, or an `Exception` for a failed row.
Returning the wrong number of results is a contract error and aborts scoring.

The existing `required` policy applies to row failures and raised batch errors:
a required term aborts scoring; an optional term contributes zero for affected
rows and records `reward/<term>/errors`. A batch error affects every row in that
batch. Terms with zero weight still execute and report metrics.

Terms without `batch_name` execute scalar calls sequentially within each fixed
batch. When fixed batching is disabled, streaming continues to call `name`, even
when `batch_name` is configured. CLAP's batch function directly evaluates the supplied batch,
bypassing its dynamic arrival queue. It retains model/device grouping and
per-row audio validation.

## Reproducibility boundary

This feature fixes the membership and order of batches delivered to reward
functions for the same input order and configured batch size. It does not make
the model batch-invariant, make stochastic preprocessing deterministic, control
rebatching inside an external inference engine, or guarantee equal scores across
devices, hardware platforms, library versions, or different batch sizes.

Continue to configure the underlying model and framework determinism policy.
`VERL_FULL_DETERMINISM` does not select CLAP's batching mode. To request fixed
single-sample scoring, set `reward.deterministic_batch_size: 1` and configure its
`batch_name` explicitly.
