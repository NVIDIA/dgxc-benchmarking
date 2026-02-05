# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""HuggingFace downloads and offline preparation.

This module provides functions to download and verify HuggingFace assets
for offline use during workload execution.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Rate limit retry defaults
_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 30
_MAX_BACKOFF_SECONDS = 300


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Check if an exception (or any exception in its chain) indicates a 429 rate limit.

    HuggingFace often wraps HTTP errors inside higher-level exceptions like
    LocalEntryNotFoundError. The top-level message may say "model or snapshot
    folder unavailable" while the actual 429 is buried in __cause__ or __context__.
    This function walks the full chain to catch those cases.
    """
    seen: Set[int] = set()
    stack: List[BaseException] = [exc]

    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))

        # Check response.status_code (requests/httpx style exceptions)
        resp = getattr(cur, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            if status == 429:
                return True

        # Check string representation as fallback
        msg = str(cur).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return True

        # Walk the exception chain
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)

    return False


def _format_exception_chain(exc: BaseException, max_depth: int = 5) -> str:
    """Format an exception chain into a compact, readable string.

    Useful for logging HuggingFace errors where the real cause (e.g., 429, 401)
    is buried under wrapper exceptions.
    """
    parts: List[str] = []
    cur: Optional[BaseException] = exc
    depth = 0
    seen: Set[int] = set()

    while cur is not None and depth < max_depth and id(cur) not in seen:
        seen.add(id(cur))

        # Include HTTP status if available
        resp = getattr(cur, "response", None)
        status = getattr(resp, "status_code", None) if resp else None
        status_str = f" [HTTP {status}]" if status else ""

        # Truncate long messages
        msg = str(cur).strip()
        if len(msg) > 200:
            msg = msg[:200] + "..."

        parts.append(f"{cur.__class__.__name__}{status_str}: {msg}" if msg else f"{cur.__class__.__name__}{status_str}")

        cur = cur.__cause__ or cur.__context__
        depth += 1

    return " → ".join(parts) if parts else "Unknown error"


def _with_retry(
    fn: Callable[[], T],
    operation: str,
    max_retries: int = _MAX_RETRIES,
    initial_backoff: int = _INITIAL_BACKOFF_SECONDS,
    max_backoff: int = _MAX_BACKOFF_SECONDS,
) -> T:
    """Execute a HuggingFace operation with retry logic for 429 rate limits.

    HuggingFace applies IP-based rate limits before checking authentication,
    so even valid tokens can hit 429 on shared IPs (e.g., cluster environments).
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise

            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(initial_backoff * (2**attempt), max_backoff)
                wait_time = round(wait_time * (0.8 + random.random() * 0.4))  # Add jitter (80-120%)
                logger.warning(
                    f"HuggingFace {operation} rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s..."
                )
                print(
                    f"  ⏳ Rate limited during {operation}. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)

    raise RuntimeError(
        f"HuggingFace {operation} failed after {max_retries} retries due to rate limiting. "
        f"Try again later or from a different network. Last error: {last_exception}"
    )


# Deterministic HuggingFace download patterns (non-weight files only)
# NOTE: *.py is required for models with custom code (e.g., DeepSeek V3, Qwen3)
# that use auto_map in config.json to reference custom Python modules like
# configuration_*.py and modeling_*.py. Without these, trust_remote_code=True
# will fail in offline mode even though trust_remote_code=False verification passes.
HF_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.yaml",
    "*.md",
    "*.py",
]

HF_IGNORE_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "*.pth",
    "*.onnx",
    "*.engine",
    "*.plan",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
]


def _normalize_hf_token(token: str) -> Optional[str]:
    """Normalize a HuggingFace token for reliable HTTP auth.

    Tokens from files or environment variables often have trailing newlines,
    whitespace, or accidental quoting that causes auth failures with cryptic
    errors like "Invalid header value".

    Returns:
        Normalized token, or None if token is empty/whitespace-only.
    """
    t = token.strip()
    # Strip common accidental quoting
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t or None


def set_hf_environment(cache_dir: str) -> None:
    """Set HuggingFace environment variables.

    CRITICAL: Must be called before any 'import transformers' or 'import huggingface_hub'
    statements, as HF_HOME is read and cached at module import time.

    Args:
        cache_dir: Base cache directory (e.g., $LLMB_INSTALL/.cache/huggingface)
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")


def build_hf_requirements_plan(workloads: Dict[str, Dict[str, Any]], selected_keys: List[str]) -> Dict[str, Set[str]]:
    """Build a per-repo HuggingFace requirements plan from workload metadata.

    Returns:
        Mapping of repo_id -> set of required assets (tokenizer/config)
    """
    requirements: Dict[str, Set[str]] = {}

    logger.debug(f"Checking {len(selected_keys)} workloads for HF downloads: {selected_keys}")

    for key in selected_keys:
        workload = workloads.get(key)
        if not workload:
            logger.warning(f"Workload '{key}' not found")
            continue

        downloads = workload.get("downloads", {}) or {}

        if not isinstance(downloads, dict):
            raise ValueError(
                f"Workload '{key}' has invalid 'downloads' section: expected dict, got {type(downloads).__name__}. "
                f"Check metadata.yaml - use proper dict structure or omit 'downloads' entirely."
            )

        has_legacy = "hf_tokenizers" in downloads
        has_new = "huggingface" in downloads
        if has_legacy and has_new:
            raise ValueError(
                f"Workload '{key}' defines both 'downloads.hf_tokenizers' and 'downloads.huggingface'. "
                f"Use only one format (legacy or new) per workload."
            )

        if has_legacy:
            tokenizers = downloads.get("hf_tokenizers", [])
            logger.debug(f"Workload '{key}': downloads.hf_tokenizers = {tokenizers}")
            if not isinstance(tokenizers, list):
                raise ValueError(
                    f"Workload '{key}' has invalid 'downloads.hf_tokenizers': expected list, got {type(tokenizers).__name__}. "
                    f"Check metadata.yaml - should be:\n"
                    f"downloads:\n"
                    f"  hf_tokenizers:\n"
                    f"    - 'model/name'"
                )
            for repo_id in tokenizers:
                if not isinstance(repo_id, str):
                    raise ValueError(
                        f"Workload '{key}' has invalid 'downloads.hf_tokenizers' entry: expected string, "
                        f"got {type(repo_id).__name__}."
                    )
                requirements.setdefault(repo_id, set()).add("tokenizer")

        if has_new:
            huggingface = downloads.get("huggingface", [])
            logger.debug(f"Workload '{key}': downloads.huggingface = {huggingface}")
            if not isinstance(huggingface, list):
                raise ValueError(
                    f"Workload '{key}' has invalid 'downloads.huggingface': expected list, got {type(huggingface).__name__}. "
                    f"Check metadata.yaml - should be:\n"
                    f"downloads:\n"
                    f"  huggingface:\n"
                    f"    - repo_id: 'model/name'\n"
                    f"      assets: [tokenizer, config]"
                )
            for item in huggingface:
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Workload '{key}' has invalid 'downloads.huggingface' entry: expected dict, "
                        f"got {type(item).__name__}."
                    )
                repo_id = item.get("repo_id")
                if not isinstance(repo_id, str) or not repo_id:
                    raise ValueError(
                        f"Workload '{key}' has invalid 'downloads.huggingface.repo_id': expected non-empty string."
                    )
                assets = item.get("assets")
                if assets is None:
                    asset_set = {"tokenizer", "config"}
                else:
                    if not isinstance(assets, list):
                        raise ValueError(
                            f"Workload '{key}' has invalid 'downloads.huggingface.assets': expected list, "
                            f"got {type(assets).__name__}."
                        )
                    if not assets:
                        raise ValueError(
                            f"Workload '{key}' has invalid 'downloads.huggingface.assets': list must not be empty."
                        )
                    asset_set = set()
                    for asset in assets:
                        if not isinstance(asset, str):
                            raise ValueError(
                                f"Workload '{key}' has invalid 'downloads.huggingface.assets' entry: expected string, "
                                f"got {type(asset).__name__}."
                            )
                        if asset not in {"tokenizer", "config"}:
                            raise ValueError(
                                f"Workload '{key}' has invalid 'downloads.huggingface.assets' value '{asset}'. "
                                f"Allowed values are 'tokenizer' and 'config'."
                            )
                        asset_set.add(asset)
                requirements.setdefault(repo_id, set()).update(asset_set)

    logger.debug(f"Total unique HF repos found: {len(requirements)}")
    return requirements


def download_huggingface_snapshots(
    requirements: Dict[str, Set[str]], cache_dir: str, token: Optional[str] = None
) -> List[str]:
    """Download HuggingFace repos deterministically (non-weight files only).

    Uses snapshot_download with fixed allow/ignore patterns. No verification is
    performed in this phase.

    Note: On failure, successfully downloaded repos remain in the cache. This is
    intentional - snapshot_download is idempotent, so retrying skips completed
    downloads and resumes faster (especially useful when rate-limited).
    """
    if not requirements:
        logger.debug("No HuggingFace repos required for download phase")
        return []

    os.makedirs(cache_dir, exist_ok=True)
    set_hf_environment(cache_dir)

    # Normalize token to handle common issues (trailing newlines, whitespace, quotes)
    # that cause cryptic "Invalid header value" errors.
    normalized_token = _normalize_hf_token(token) if token else None

    if token and not normalized_token:
        logger.warning("HF token provided but empty after normalization - treating as no token")

    if normalized_token:
        # Set HF_TOKEN env var for downstream code (e.g., transformers.AutoTokenizer)
        # that uses implicit token resolution. The explicit token= parameter we pass
        # to snapshot_download takes precedence, but this covers other HF calls.
        os.environ["HF_TOKEN"] = normalized_token
        logger.debug(f"Authenticating with HuggingFace (token: {_obfuscate_token(normalized_token)})")
    else:
        logger.debug("No HF token provided - downloads may be rate limited and gated repos will fail")

    # Import HF libs only after HF_HOME/HF_HUB_CACHE/HF_TOKEN are set.
    from huggingface_hub import snapshot_download

    repos = sorted(requirements.keys())

    print("\nDownloading HuggingFace files")
    print("--------------------------------")
    print("\nRequired HuggingFace repos:")
    for repo_id in repos:
        assets = ", ".join(sorted(requirements[repo_id]))
        suffix = f" ({assets})" if assets else ""
        print(f"  - {repo_id}{suffix}")

    print("\nDownloading non-weight files...")
    print("(Progress bars are expected)")

    successful = []
    total = len(repos)
    for idx, repo_id in enumerate(repos, 1):
        logger.debug(f"Snapshotting HuggingFace repo {idx}/{total}: {repo_id}")
        print(f"\n[{idx}/{total}] {repo_id}")
        try:
            snapshot_path = _with_retry(
                lambda rid=repo_id, tok=normalized_token: snapshot_download(
                    repo_id=rid,
                    allow_patterns=HF_ALLOW_PATTERNS,
                    ignore_patterns=HF_IGNORE_PATTERNS,
                    token=tok,
                ),
                operation=f"snapshot '{repo_id}'",
            )
        except Exception as exc:
            # Provide actionable error: HF often wraps 401/403/429 in generic errors
            chain = _format_exception_chain(exc)
            logger.error("HuggingFace snapshot failed for '%s': %s", repo_id, chain)
            raise RuntimeError(f"HuggingFace snapshot failed for '{repo_id}'. Error chain: {chain}") from exc
        _maybe_inject_nemotron_config(
            repo_id=repo_id,
            assets=requirements[repo_id],
            snapshot_path=snapshot_path,
        )
        # Tokenizer finalization may download additional files (e.g., custom code)
        # and can hit rate limits independently of snapshot_download.
        _with_retry(
            lambda rid=repo_id, tok=normalized_token, path=snapshot_path: _maybe_finalize_tokenizer_snapshot(
                repo_id=rid,
                assets=requirements[rid],
                snapshot_path=path,
                token=tok,
            ),
            operation=f"tokenizer finalization '{repo_id}'",
        )
        successful.append(repo_id)

    print(f"\nSuccessfully downloaded {len(successful)} repo(s).")
    print("Download phase complete.")
    return successful


def _verify_hf_asset(repo_id: str, asset: str, load_fn: Any) -> None:
    """Verify a HuggingFace asset loads locally.

    Uses trust_remote_code=True to match runtime behavior. This is safe because:
    1. Only curated repos from metadata.yaml are downloaded
    2. Runtime (Megatron-Bridge) already uses trust_remote_code=True
    3. Models like DeepSeek V3 and Qwen3 require custom Python modules
    """
    try:
        load_fn(
            repo_id,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.debug(
            "Offline verification failed for repo '%s' (asset: %s): %s",
            repo_id,
            asset,
            exc,
            exc_info=True,
        )
        reason = f"{exc.__class__.__name__}: {exc}" if str(exc) else exc.__class__.__name__
        raise RuntimeError(
            f"HuggingFace offline verification failed for repo '{repo_id}' (asset: {asset}). "
            f"Reason: {reason}. Ensure required files (including *.py for custom models) "
            f"are present in the local HF cache."
        ) from exc


def _maybe_inject_nemotron_config(repo_id: str, assets: Set[str], snapshot_path: str) -> None:
    """Inject minimal config.json for tokenizer-only Nemotron repos when needed."""
    if "tokenizer" not in assets or "config" in assets:
        return

    if "nemotron" not in repo_id.lower():
        return

    config_path = Path(snapshot_path) / "config.json"
    if config_path.exists():
        return

    logger.debug("Injecting minimal config.json for tokenizer-only Nemotron repo '%s'", repo_id)
    with open(config_path, "w") as config_file:
        json.dump({"model_type": "nemotron"}, config_file, indent=2)

    print("  → Injected minimal config.json for tokenizer-only Nemotron")


def _maybe_finalize_tokenizer_snapshot(
    repo_id: str, assets: Set[str], snapshot_path: str, token: Optional[str]
) -> None:
    """Materialize tokenizer metadata files for repos that don't ship them."""
    if "tokenizer" not in assets:
        return

    snapshot_root = Path(snapshot_path)
    tokenizer_config_path = snapshot_root / "tokenizer_config.json"
    special_tokens_path = snapshot_root / "special_tokens_map.json"
    if tokenizer_config_path.exists() and special_tokens_path.exists():
        return

    from transformers import AutoTokenizer

    logger.debug(
        "Generating tokenizer metadata for repo '%s' (missing tokenizer_config.json or special_tokens_map.json)",
        repo_id,
    )

    def _load_tokenizer(trust_remote_code: bool) -> Any:
        return AutoTokenizer.from_pretrained(
            repo_id,
            token=token,
            trust_remote_code=trust_remote_code,
        )

    try:
        tokenizer = _load_tokenizer(trust_remote_code=False)
    except Exception as exc:
        # Only log full traceback for non-rate-limit errors (rate limits will retry)
        is_rate_limit = _is_rate_limit_error(exc)
        logger.debug(
            "Tokenizer load failed for repo '%s' with trust_remote_code=False: %s",
            repo_id,
            exc,
            exc_info=not is_rate_limit,
        )
        # NOTE: At time of writing, this edge case has only been observed for Nemotron repos.
        if "nemotron" not in repo_id.lower():
            raise
        logger.debug(
            "Retrying tokenizer load for Nemotron repo '%s' with trust_remote_code=True",
            repo_id,
        )
        tokenizer = _load_tokenizer(trust_remote_code=True)

    tokenizer.save_pretrained(snapshot_path)
    print("  → Generated tokenizer metadata files")


def verify_huggingface_assets(requirements: Dict[str, Set[str]], cache_dir: str) -> None:
    """Verify required HuggingFace assets load offline using local cache only."""
    if not requirements:
        logger.debug("No HuggingFace repos required for verification phase")
        return

    os.makedirs(cache_dir, exist_ok=True)
    set_hf_environment(cache_dir)

    # Import HF libs only after HF_HOME/HF_HUB_CACHE are set.
    from transformers import AutoConfig, AutoTokenizer

    repos = sorted(requirements.keys())

    print("\nVerifying HuggingFace assets (offline)")
    print("-------------------------------------")
    for idx, repo_id in enumerate(repos, 1):
        assets = requirements[repo_id]
        asset_list = ", ".join(sorted(assets))
        suffix = f" ({asset_list})" if asset_list else ""
        print(f"\n[{idx}/{len(repos)}] {repo_id}{suffix}")

        if "tokenizer" in assets:
            _verify_hf_asset(repo_id, "tokenizer", AutoTokenizer.from_pretrained)
            print("  ✓ tokenizer")

        if "config" in assets:
            _verify_hf_asset(repo_id, "config", AutoConfig.from_pretrained)
            print("  ✓ config")

    print(f"\nSuccessfully verified {len(repos)} repo(s).")


def download_huggingface_files_for_workloads(
    workloads: Dict[str, Dict[str, Any]], selected_keys: List[str], install_path: str, hf_token: Optional[str] = None
) -> None:
    """Download and verify HuggingFace assets required by selected workloads.

    This entrypoint builds a requirements plan, runs the download phase, then
    verifies assets offline. It supports legacy downloads.hf_tokenizers and the
    new downloads.huggingface schema.
    """
    requirements = build_hf_requirements_plan(workloads, selected_keys)
    if not requirements:
        logger.debug("No HuggingFace assets required")
        return

    hf_cache_dir = os.path.join(install_path, ".cache", "huggingface")
    os.makedirs(hf_cache_dir, exist_ok=True)

    download_huggingface_snapshots(requirements, hf_cache_dir, hf_token)
    verify_huggingface_assets(requirements, hf_cache_dir)


def _obfuscate_token(token: str) -> str:
    """Obfuscate token for safe logging (show first 6 and last 4 characters)."""
    if len(token) <= 12:
        return token[:2] + "*" * (len(token) - 2)
    return f"{token[:6]}...{token[-4:]}"
