local num_episodes_per_iteration = 1024;
// local num_rollouts_per_sample = 32;
local num_dataset_samples_per_iteration = 16;
local total_num_iterations = 1000;

// NOTE: upstream imported 'polIter_deepseekSft2_vineppo_MATH.jsonnet', which is NOT
// shipped in the SPO release (the import fails) and, being VinePPO-based, lacks the
// SPO probability-mask update. Repointed to the shipped, working SPO-chain base so
// this is a genuine SPO-tree config (use_prob_mask=true), mirroring the structure of
// polIter_qwen1b_spo_tree_MATH.jsonnet.
(import 'polIter_deepseekSft2_spo_chain_MATH.jsonnet')
+ {
  episode_generator+: {
    type: 'hybrid_episode_generator',
    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
    inference_strategy+: {
      type: 'hybrid',
      M: 66,
      max_depth: 3,
      branch_factor_strategy: {
        type: 'list',
        branch_factors: [
          { depth: 0, branch_factor: 6 },
          { depth: 1, branch_factor: 6 },
          { depth: 2, branch_factor: 6 },
        ],
      },
    },
    num_episodes_per_iteration: num_episodes_per_iteration,

  },
  num_episodes_per_iteration: num_episodes_per_iteration,

  trainer+: {
    num_epochs_per_iteration: 1,
    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 8,
      per_device_eval_batch_size: 2,
    },
  },
}
