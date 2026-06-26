{
    type: 'mean_at_k',

    // grpocredit/verl key namespace. Final wandb keys look like
    // `val-core/math_v2/reward/mean@16` so they overlay grpocredit exactly.
    data_source: 'math_v2',
    var_name: 'reward',

    // Bootstrap settings ported from verl (best/worst@k). Defaults match verl.
    n_bootstrap: 1000,
    seed: 42,

    // Must be set by the importing config (used by the model guard). Only
    // deepseekmath-on-math_v2 is supported; qwen / R1-Distill raise NotImplementedError.
    model_name_or_path: null,
}
