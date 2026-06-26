import os

import diskcache
import platformdirs

from guidance.llms.caches import Cache


class DiskCache(Cache):
    """DiskCache is a cache that uses diskcache lib."""
    def __init__(self, llm_name: str):
        # The cache key is only the LLM *class* name (e.g. "openai_vllm"), so by
        # default every rank of every concurrent job shares one SQLite db under
        # ~/.cache/guidance on the (network) home FS. Concurrent multi-node writers
        # corrupt SQLite there ("database disk image is malformed" / "file is not a
        # database"). Honor GUIDANCE_CACHE_DIR so launchers can pin a per-job,
        # node-local dir; falls back to the original platformdirs path when unset.
        base_dir = os.environ.get("GUIDANCE_CACHE_DIR") or platformdirs.user_cache_dir(
            "guidance"
        )
        self._diskcache = diskcache.Cache(
            os.path.join(base_dir, f"_{llm_name}.diskcache")
        )

    def __getitem__(self, key: str) -> str:
        return self._diskcache[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._diskcache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._diskcache
    
    def clear(self):
        self._diskcache.clear()
