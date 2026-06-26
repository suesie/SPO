(import 'openai_vllm.jsonnet') + {
  // Local snapshot path, NOT the repo-id — see deepseekmath7b-sft-MATH-v2.jsonnet:
  // vLLM 0.4.0 would otherwise call the Hub (HfFileSystem.ls) and crash under
  // HF_HUB_OFFLINE=1. Used by the in-loop MATH-validation eval pipeline.
  model: '/lustre-storage/datasets/zengh/huggingface/hub/models--realtreetune--deepseekmath-7b-sft-MATH-v2/snapshots/8b387c255b3bfaaaef2e650d56fecfde1c56ea96',
  tokenizer_name: '/lustre-storage/datasets/zengh/huggingface/hub/models--realtreetune--deepseekmath-7b-sft-MATH-v2/snapshots/8b387c255b3bfaaaef2e650d56fecfde1c56ea96',
}
