# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""HuggingFace tokenizer download and offline preparation.

This module provides functions to download and prepare HuggingFace tokenizers
for offline use during workload execution.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)

# File patterns for tokenizer downloads via snapshot_download
TOKENIZER_FILE_PATTERNS = [
    "*.model",  # SentencePiece models (LLaMA, Nemotron)
    "*.yaml",  # Config files
    "*.json",  # Tokenizer configs
    "*.txt",  # Vocab files
    "*.py",  # Custom tokenizer code
    "README.md",
]

# Exclude model weights and large binaries
EXCLUDE_PATTERNS = [
    "*.pt",
    "*.safetensors",
    "*.bin",
    "*.msgpack",
    "*.h5",
    "*.ot",
]


def set_hf_environment(cache_dir: str) -> None:
    """Set HuggingFace environment variables.

    CRITICAL: Must be called before any 'import transformers' or 'import huggingface_hub'
    statements, as HF_HOME is read and cached at module import time.

    Args:
        cache_dir: Base cache directory (e.g., $LLMB_INSTALL/.cache/huggingface)
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")


def get_required_tokenizers(workloads: Dict[str, Dict[str, Any]], selected_keys: List[str]) -> List[str]:
    """Extract unique tokenizer model IDs from selected workloads.

    Returns:
        Deduplicated sorted list of HuggingFace model IDs
    """
    tokenizer_set = set()

    logger.debug(f"Checking {len(selected_keys)} workloads for tokenizers: {selected_keys}")

    for key in selected_keys:
        workload = workloads.get(key)
        if not workload:
            logger.warning(f"Workload '{key}' not found")
            continue

        downloads = workload.get('downloads', {})

        if not isinstance(downloads, dict):
            raise ValueError(
                f"Workload '{key}' has invalid 'downloads' section: expected dict, got {type(downloads).__name__}. "
                f"Check metadata.yaml - use proper dict structure or omit 'downloads' entirely."
            )

        tokenizers = downloads.get('hf_tokenizers', [])
        logger.debug(f"Workload '{key}': downloads.hf_tokenizers = {tokenizers}")

        if not isinstance(tokenizers, list):
            raise ValueError(
                f"Workload '{key}' has invalid 'downloads.hf_tokenizers': expected list, got {type(tokenizers).__name__}. "
                f"Check metadata.yaml - should be:\n"
                f"downloads:\n"
                f"  hf_tokenizers:\n"
                f"    - 'model/name'"
            )

        tokenizer_set.update(tokenizers)

    logger.debug(f"Total unique tokenizers found: {len(tokenizer_set)}")
    return sorted(tokenizer_set)


def verify_offline_tokenizer(model_id: str, token: Optional[str] = None) -> bool:
    """Verify tokenizer can be loaded in offline mode."""
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            trust_remote_code=True,
            token=token,
        )
        return True
    except Exception:
        return False


def prepare_tokenizer_for_offline(model_id: str, token: Optional[str] = None) -> None:
    """Prepare tokenizer for offline use with two-phase approach.

    Approach 1: AutoTokenizer.from_pretrained() - standard download with auto-caching
    Approach 2: snapshot_download() - explicit file download + config injection for broken repos

    IMPORTANT: Do NOT pass cache_dir parameter to HuggingFace functions.
               Let them use HF_HOME/HF_HUB_CACHE environment variables instead.
               Passing cache_dir bypasses the standard hub/ subdirectory structure.

    Raises:
        Exception: If tokenizer cannot be prepared for offline use
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Approach 1: Standard download
    try:
        logger.debug("  → Attempting AutoTokenizer.from_pretrained()")
        AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
        )

        logger.debug("  → Verifying offline access...")
        if verify_offline_tokenizer(model_id, token):
            logger.debug("  ✓ Offline verification passed")
            print("  ✓ Ready for offline use")
            return

        # Cache incomplete - need to fetch missing files
        logger.debug("  → Cache incomplete, fetching with snapshot_download()")

    except Exception as e:
        logger.debug(f"  ✗ AutoTokenizer.from_pretrained() failed: {e}")
        logger.debug("  → Falling back to snapshot_download() with explicit file patterns")

    # Approach 2: Explicit file download for broken repos or cache issues
    snapshot_path = snapshot_download(
        repo_id=model_id,
        token=token,
        allow_patterns=TOKENIZER_FILE_PATTERNS,
        ignore_patterns=EXCLUDE_PATTERNS,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(snapshot_path)

    # Inject config.json for repos that lack it (known Nemotron issue)
    config_path = Path(snapshot_path) / "config.json"
    if not config_path.exists():
        logger.debug("  → Injecting missing config.json...")
        print("  → Injecting missing config.json", end="", flush=True)

        if "nemotron" in model_id.lower():
            model_type = "nemotron"  # Known issue, upstream ticket filed
        else:
            raise ValueError(
                f"Model '{model_id}' requires config.json but repo is missing it. "
                f"This is a known issue for Nemotron models. If this is a new model, "
                f"investigate the upstream repo and add explicit handling."
            )

        with open(config_path, "w") as f:
            json.dump({"model_type": model_type}, f, indent=2)
        logger.debug("  → Created config.json")
        print(" (repo missing config)")

    logger.debug("  → Verifying offline access...")
    if not verify_offline_tokenizer(model_id, token):
        raise RuntimeError("Offline verification failed - tokenizer cannot be loaded with local_files_only=True")

    logger.debug("  ✓ Offline verification passed")
    print("  ✓ Ready for offline use")


def _obfuscate_token(token: str) -> str:
    """Obfuscate token for safe logging (show first 6 and last 4 characters)."""
    if len(token) <= 12:
        return token[:2] + "*" * (len(token) - 2)
    return f"{token[:6]}...{token[-4:]}"


def download_tokenizers(tokenizers: List[str], token: Optional[str] = None) -> List[str]:
    """Download and prepare multiple tokenizers for offline use.

    Returns:
        List of successfully prepared tokenizer model IDs

    Raises:
        Exception: If any tokenizer fails to download or prepare
    """
    from huggingface_hub import login

    # Login to HuggingFace Hub once before downloading all tokenizers
    # This sets the token globally for all HF operations, which is more reliable
    # than passing token= to individual API calls
    if token:
        logger.debug(f'Authenticating with HuggingFace (token: {_obfuscate_token(token)}, obfuscated for security)')
        login(token=token, add_to_git_credential=False)
    else:
        logger.debug('No HF_TOKEN provided - downloads may be rate limited')

    successful = []
    total = len(tokenizers)
    for idx, model_id in enumerate(tokenizers, 1):
        logger.debug(f'Processing tokenizer {idx}/{total}: {model_id}')
        print(f"\n[{idx}/{total}] {model_id}")
        prepare_tokenizer_for_offline(model_id, token)
        successful.append(model_id)
    return successful


def fetch_hf_tokenizers_for_workloads(
    workloads: Dict[str, Dict[str, Any]], selected_keys: List[str], install_path: str, hf_token: Optional[str] = None
) -> None:
    """Fetch HuggingFace tokenizers required by selected workloads.

    Main entry point for tokenizer downloads. Extracts tokenizers from workload
    metadata, sets HF environment, and downloads for offline use.

    Raises:
        Exception: If any required tokenizer fails to download

    Note:
        On first run, transformers library may print an informational message about
        missing ML frameworks (PyTorch/TensorFlow). This is expected and harmless -
        only tokenizer functionality is needed, not full model support.
    """
    required_tokenizers = get_required_tokenizers(workloads, selected_keys)

    if not required_tokenizers:
        logger.debug("No tokenizers required")
        return

    # Set HF environment before any HF library imports
    hf_cache_dir = os.path.join(install_path, ".cache", "huggingface")
    os.makedirs(hf_cache_dir, exist_ok=True)
    set_hf_environment(hf_cache_dir)

    # Print header and list all required tokenizers upfront (like images)
    print("\nDownloading HuggingFace tokenizers")
    print("-----------------------------------")
    print("\nRequired tokenizers:")
    for model_id in required_tokenizers:
        print(f"  - {model_id}")

    print("\nDownloading tokenizer files...")
    print("(Framework warnings and file progress bars are expected)")

    # Download all tokenizers (will raise on failure)
    # Library will show file-level progress bars during download
    successful = download_tokenizers(required_tokenizers, hf_token)

    # Print final summary
    print(f"\nSuccessfully prepared {len(successful)} tokenizer(s) for offline use.")
