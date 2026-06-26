(import 'openai_vllm.jsonnet') + {
  // Local snapshot path, NOT the repo-id: vLLM 0.4.0 resolves a repo-id via
  // HfFileSystem.ls() (a Hub API call) which raises OfflineModeIsEnabled under
  // HF_HUB_OFFLINE=1. A local dir is loaded directly (is_local=True), so vLLM
  // generation works fully offline. (transformers-side hf_model_name can stay a
  // repo-id since from_pretrained honors the offline cache.)
  model: '/lustre-storage/datasets/zengh/huggingface/hub/models--realtreetune--deepseekmath-7b-sft-MATH-v2/snapshots/8b387c255b3bfaaaef2e650d56fecfde1c56ea96',
}
