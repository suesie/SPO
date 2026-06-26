// Reset episode_generator.max_question_length to the UPSTREAM value (200).
//
// WHY THIS EXISTS:
//   polIter_qwen1b_spo_chain_MATH.jsonnet (the shared parent of the qwen1b grpo /
//   spo_chain / spo_tree configs) has a LOCAL, UNCOMMITTED edit that raises
//   max_question_length 200 -> 512. Upstream AIFrameResearch/SPO@1e64f0c ships 200
//   (verified via `git blame`). 512 vs 200 changes which MATH-train questions pass the
//   length filter (~12% of train), so it is a real TRAINING-DATA delta.
//
//   Layer this file LAST (via the launcher's EXTRA_CONFIGS) to reproduce the official
//   upstream training-data selection WITHOUT reverting the working-tree edit that other
//   local runs depend on. treetune merges --configs with jsonnet `+`, so the scalar
//   below overrides the inherited 512.
//
// NOTE: This intentionally touches ONLY max_question_length. The other working-tree
//   deltas are training-equivalent: model path is a content-identical local snapshot
//   (HF refs/main == ad9f0ae0), the in-memory dataset.shuffle uses the same seed (=> same
//   permutation), and save_steps/checkpoint_keep_steps affect checkpoint cadence only.
{
  episode_generator+: {
    max_question_length: 200,
  },
}
