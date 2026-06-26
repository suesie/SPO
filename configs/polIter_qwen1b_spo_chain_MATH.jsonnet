// Local snapshot dir (NOT repo-ID): under HF_HUB_OFFLINE=1, vLLM 0.4 resolves a
// repo-ID via a Hub API call that raises OfflineModeIsEnabled, crashing all
// per-rank servers at iter-0 (see exp_track.md root-cause for the 7B run).
local hf_model_name = '/lustre-storage/datasets/zengh/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562';

local math_task = (import 'tasks/math_inplace_no_answer_prefix.jsonnet') + {
  prepend_in_context_few_shot: false,
  ensure_fit_in_context_size: false,
};
local num_episodes_per_iteration = 512;
local num_rollouts_per_sample = 8;
local num_dataset_samples_per_iteration = num_episodes_per_iteration / num_rollouts_per_sample;
local total_num_iterations = 1000;

local sampling_temperature = 0.6;

local num_mc_rollouts = 4;


(import 'gvar.jsonnet')
+ (import 'prompt_library/qwen_MATH.jsonnet')
+ (import 'runtimes/policy_iteration.jsonnet')
+ (import 'trainers/ppo_MATH.jsonnet')
+ {
  episode_generator+: {
    type: 'math_episode_generator_w_mc_advantages',
    // Override the task
    task: math_task,

    reward_function: {
      type: 'math_reward_function',
      penalize_unfinished_response: true,
      unfinished_response_penalty: 0.0,
      math_task: $.episode_generator.task,
    },

    initial_model_name_or_path: hf_model_name,

    dataset_num_samples_per_iteration: num_dataset_samples_per_iteration,
    total_num_iterations: $.num_iterations,

    vllm_gpu_memory_utilization: 'auto',
    vllm_min_available_gpu_memory_mb: 4 * 1024,
    wait_until_memory_release: true,
    vllm_server+: {
      swap_space: 32,
      max_num_seqs: 512,
      enable_prefix_caching: true,
    },

    max_sequence_length: null,
    // BUGFIX (local): upstream shipped 200, which is the GSM8K value (see
    // polIter_qwen05b_spo_chain_gsm8k.jsonnet) carried over to MATH. It drops ~12% of
    // MATH train (templated prompt >200 tok) since the ~85-tok instruction eats into the
    // budget. Every other MATH config is larger: base default 1512
    // (episode_generators/math_episode_generator.jsonnet), 7B inherits 1512,
    // polIter_qwen1_5b_base_spo_chain_MATH.jsonnet uses 512. Set to 512 to match the base
    // qwen-1.5B MATH config.
    max_question_length: 512,
    question_sampler: {
      type: 'random',  // not used
    },

    append_bos_to_query: false,
    append_eos_to_response: false,

    dataset_shuffle_on_each_iteration: true,
    dataset_shuffle_before_portion: true,
    dataset_sample_with_replacement: true,
    fill_missing_episodes: true,

    // cutpoint_interval: 40,

    question_template: $.prompt_library.tree.question_template,

    inference_strategy: {
      type: 'cot',

      max_concurrent_programs: 128,
      max_concurrent_generations: 128,

      samples: num_rollouts_per_sample,
      max_depth: 100,  // Deprecated parameter. Doesn't do anything.

      node_expander: {
        type: 'efficient_iid',
        program: $.prompt_library.tree.expansion.iid,
        program_kwargs+: {
          temperature: sampling_temperature,
          top_p: 1,
          max_tokens: 4096,  // Long CoT
          stop: '"\n\n\nProblem:"',  // not used
          logprobs: 0,
        },
        node_text_template: '{chain_of_thought}',

        // Needed to compute max_tokens on the fly
        model_context_size: 2048,  // Long CoT
        tokenizer: $.tokenizer,
      },

      answer_extractor: {
        type: 'identity',
        node_key_name: 'text',
      },

      guidance_llm: (import 'guidance_llms/qwen1b.jsonnet') + { api_base: 'none' },

      question_field: 'query',
      question_template: $.prompt_library.tree.question_template,

      no_cache: true,
    },

    value_estimation_inference_strategy+: {
      type: 'cot',

      max_concurrent_programs: 512,
      max_concurrent_generations: 512,

      samples: num_mc_rollouts,
      max_depth: 100,  // Deprecated parameter. Doesn't do anything.

      node_expander: $.episode_generator.inference_strategy.node_expander,
      answer_extractor: {
        type: 'identity',
        node_key_name: 'text',
      },

      guidance_llm: $.episode_generator.inference_strategy.guidance_llm,

      question_field: 'query',
      question_template: '{query}',

      no_cache: true,
    },
  },

  tokenizer: {
    type: 'pretrained',
    hf_model_name: $.episode_generator.initial_model_name_or_path,
  },
  use_deepspeed: true,

  num_iterations: total_num_iterations,
  num_episodes_per_iteration: num_episodes_per_iteration,
  episodes_cloud_log_steps: 50,


  trainer+: {
    params+: {
      temperature: $.episode_generator.inference_strategy.node_expander.program_kwargs.temperature,
      use_prob_mask: true,
    },

    num_epochs_per_iteration: 1,

    actor_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

    critic_model: null,
    critic_deepspeed_config: null,
    save_hf_critic_checkpoint: false,

    reference_model+: { hf_model_name: $.episode_generator.initial_model_name_or_path },

    actor_deepspeed_config: (import 'deepspeed/zero_0.jsonnet'),
    move_reference_model_to_cpu: true,

    // To prevent OOM errors
    report_entropy: false,

    general_training_args+: {
      target_train_batch_size: 128,
      per_device_train_batch_size: 2,
      per_device_eval_batch_size: 2,
      gradient_accumulation_steps: null,  // Will be auto computed
      // Checkpoint at 25,50,...,1000 (match the 7B run's cadence).
      // checkpoint_keep_steps is NOT a count: clean_checkpoints retains every
      // checkpoint whose iter % checkpoint_keep_steps == 0 (modulus). With
      // save_steps == checkpoint_keep_steps == 25 → keep iters 25,50,...,1000 (40 ckpts).
      save_steps: 25,
      checkpoint_keep_steps: 25,
    },

  },
}
+ (import 'qwen1b_for_MATH_eval.jsonnet')
+ (import 'trainers/lam1.jsonnet')
+ (import 'trainers/refKl0.0001.jsonnet')
+ (import 'trainers/klLoss.jsonnet')
