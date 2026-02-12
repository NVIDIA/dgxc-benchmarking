#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

# Configuration
readonly MIN_PYTHON_VERSION="3.12"
readonly RECOMMENDED_PYTHON_VERSION="3.12"
readonly PINNED_UV_VERSION="0.9.28"

# Welcome banner
echo "════════════════════════════════════════════════════════════════════"
echo "🚀 LLM Benchmarking Collection - Quick Setup"
echo "════════════════════════════════════════════════════════════════════"
echo "This script will:"
echo "  • Set up a Python $RECOMMENDED_PYTHON_VERSION virtual environment (if needed)"
echo "  • Install essential tools (llmb-run, llmb-install)"
echo "  • Launch the main installer for benchmark configurations"
echo ""
echo "Note: Benchmark recipes require Python $RECOMMENDED_PYTHON_VERSION"
echo ""

# State for summary
UV_INSTALLED_NOW=false
UV_WAS_ADDED_TO_PATH=false
CREATED_VENV=false
VENV_DIR=""
MULTI_UV_WARNING_SHOWN=false

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Prefer user-level uv install path when present.
ensure_local_uv_path() {
    local local_uv="$HOME/.local/bin/uv"
    if [[ -x $local_uv ]]; then
        case ":$PATH:" in
            *":$HOME/.local/bin:"*) ;;
            *)
                export PATH="$HOME/.local/bin:$PATH"
                UV_WAS_ADDED_TO_PATH=true
                ;;
        esac
    fi
}

# Get uv version string (e.g., 0.9.28)
get_uv_version() {
    uv --version 2> /dev/null | awk '{print $2}'
}

# Returns success (0) if version A is greater than version B.
version_gt() {
    [[ $1 =~ ^[0-9] ]] && [[ $2 =~ ^[0-9] ]] \
        && [[ $1 != "$2" ]] && [[ "$(printf '%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]
}

# Warn if multiple uv binaries are available in PATH.
# This warning is intentionally suppressed in low-risk cases to reduce noise.
warn_if_multiple_uv_installs() {
    if [[ $MULTI_UV_WARNING_SHOWN == true ]] || ! command_exists uv; then
        return 0
    fi

    local uv_paths
    uv_paths=$(type -ap uv 2> /dev/null | awk '!seen[$0]++')

    local count
    count=$(printf "%s\n" "$uv_paths" | sed '/^$/d' | wc -l | tr -d ' ')
    if [[ $count -gt 1 ]]; then
        local active_uv
        active_uv=$(command -v uv)
        local managed_uv="$HOME/.local/bin/uv"

        # Intentionally skip warning when managed uv is already first in PATH
        # and we did not need to modify PATH in this session.
        if [[ $active_uv == "$managed_uv" && $UV_WAS_ADDED_TO_PATH != true ]]; then
            MULTI_UV_WARNING_SHOWN=true
            return 0
        fi

        local active_version
        active_version=$(get_uv_version 2> /dev/null || echo "unknown")
        local version_note="v${active_version}"
        if [[ $active_uv == "$managed_uv" ]]; then
            version_note="v${active_version} ✅"
        fi

        echo ""
        echo "⚠️  Multiple uv binaries found in PATH:"
        while IFS= read -r uv_path; do
            [[ -n $uv_path ]] && echo "   - $uv_path"
        done <<< "$uv_paths"
        echo ""
        echo "   Active: $active_uv ($version_note)"
        echo ""
        echo "   If you run llmb-install in a new shell, a different uv may be"
        echo "   resolved. To avoid issues, ensure ~/.local/bin appears first"
        echo "   in your PATH (e.g., in ~/.bashrc)."
        echo ""
    fi

    MULTI_UV_WARNING_SHOWN=true
}

# Check if Python version meets minimum requirement
check_python_version() {
    local python_cmd="$1"
    # Validate MIN_PYTHON_VERSION format (e.g., "3.12")
    if [[ ! $MIN_PYTHON_VERSION =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "ERROR: Invalid MIN_PYTHON_VERSION format: $MIN_PYTHON_VERSION" >&2
        return 1
    fi
    local major="${MIN_PYTHON_VERSION%%.*}"
    local minor="${MIN_PYTHON_VERSION#*.}"
    if ! $python_cmd -c "import sys; exit(0 if sys.version_info >= ($major, $minor) else 1)" 2> /dev/null; then
        return 1
    fi
    return 0
}

# Get Python version string
get_python_version() {
    local python_cmd="$1"
    $python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2> /dev/null || echo "unknown"
}

# Check if we're in a virtual environment (standard venv/virtualenv or conda)
in_virtual_env() {
    [[ -n ${VIRTUAL_ENV:-} ]] || [[ -n ${CONDA_DEFAULT_ENV:-} ]]
}

# Reusable message functions
show_uv_benefits() {
    echo "⚡ uv - Recommended Python Package Manager"
    echo ""
    echo "  • Makes benchmark installations 2-5x faster"
    echo "  • Automatically installs Python $RECOMMENDED_PYTHON_VERSION (required for recipes)"
    echo "  • Lightweight tool (~10MB) with better dependency resolution"
    echo "  • This release supports uv versions up to $PINNED_UV_VERSION for recipe compatibility"
    echo ""
    echo "⚠️ Note: Will download installer from https://astral.sh/uv (official source)"
}

show_python_error() {
    local current_version=$(get_python_version python3)
    echo "❌ Cannot proceed: Python $current_version found (requires $MIN_PYTHON_VERSION)"
    echo ""
    echo "Options:"
    echo "  1. Install 'uv $PINNED_UV_VERSION' (recommended) - automatically installs Python $RECOMMENDED_PYTHON_VERSION"
    echo "  2. Use conda to create a Python $RECOMMENDED_PYTHON_VERSION environment"
    echo "  3. Use pyenv or upgrade system Python to $RECOMMENDED_PYTHON_VERSION+"
    echo ""
    echo "Re-run this script after installing one of the above."
    exit 1
}

# Install pinned uv version required by recipes in this release
install_uv() {
    echo "Installing uv $PINNED_UV_VERSION..."
    if ! curl -LsSf "https://astral.sh/uv/${PINNED_UV_VERSION}/install.sh" | sh; then
        echo "❌ uv installation failed"
        return 1
    fi

    ensure_local_uv_path
    hash -d uv 2> /dev/null || true

    # Check if uv is now available and matches the pinned version
    if command_exists uv; then
        local uv_version
        uv_version=$(get_uv_version || true)
        if [[ $uv_version == "$PINNED_UV_VERSION" ]]; then
            echo "✅ uv $PINNED_UV_VERSION installed"
            UV_INSTALLED_NOW=true
            warn_if_multiple_uv_installs
            return 0
        fi
    fi

    echo "❌ uv installation failed: expected $PINNED_UV_VERSION, found $(get_uv_version || echo unknown)"
    return 1
}

# Ensure uv is compatible for this release.
# We only correct versions newer than the known-compatible target.
ensure_compatible_uv() {
    ensure_local_uv_path

    if ! command_exists uv; then
        return 1
    fi

    local uv_version
    uv_version=$(get_uv_version || true)
    if [[ -z $uv_version ]]; then
        echo "⚠️  Unable to determine uv version. Installing uv $PINNED_UV_VERSION..."
        install_uv
        return $?
    fi

    if version_gt "$uv_version" "$PINNED_UV_VERSION"; then
        echo "⚠️  Detected uv $uv_version, but this release supports up to uv $PINNED_UV_VERSION."
        echo "   Installing uv $PINNED_UV_VERSION for compatibility..."
        install_uv
        return $?
    fi

    warn_if_multiple_uv_installs
    return 0
}

# Environment setup with proper sequencing
setup_environment() {
    echo "🔍 Checking Python environment..."

    # Check if we're already in a good virtual environment
    if in_virtual_env && check_python_version python3; then
        local py_ver=$(get_python_version python3)
        echo "✅ Using existing virtual environment with Python $py_ver"

        # Offer uv installation for faster benchmark installs (optional for existing good venvs)
        if ! command_exists uv; then
            echo ""
            echo "💡 Optional: Install 'uv $PINNED_UV_VERSION' for 2-5x faster benchmark installs"
            echo "   (Your current environment is ready and will work fine)"
            echo "   Downloads from https://astral.sh/uv (official source)"
            echo ""
            read -p "Install 'uv $PINNED_UV_VERSION' (recommended)? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_uv
            fi
        elif ! ensure_compatible_uv; then
            echo "❌ Cannot continue without a compatible uv installation."
            exit 1
        fi
        return 0
    fi

    # Need to create a new virtual environment
    echo "📦 Setting up new virtual environment..."

    # If uv is available, use it (can handle any Python version)
    if command_exists uv; then
        if ! ensure_compatible_uv; then
            echo "❌ Cannot continue without a compatible uv installation."
            exit 1
        fi
        echo "Using uv to create virtual environment with Python $RECOMMENDED_PYTHON_VERSION..."
        create_venv_with_uv
        return 0
    fi

    # No uv available - check system Python version
    local sys_python_ver=$(get_python_version python3)

    if check_python_version python3; then
        # System Python is 3.12+ - offer choice
        echo ""
        echo "💡 Your system has Python $sys_python_ver"
        echo ""
        show_uv_benefits
        echo ""
        echo "Or use your system Python $sys_python_ver (will work, but slower installs)"
        echo ""
        read -p "Install 'uv $PINNED_UV_VERSION'? [Y/n]: " -n 1 -r
        echo

        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            if install_uv; then
                create_venv_with_uv
                return 0
            else
                echo "⚠️  uv installation failed, falling back to system Python..."
            fi
        fi

        # Use system Python
        echo "📦 Creating virtual environment with system Python $sys_python_ver..."
        create_venv_with_python
        return 0
    else
        # System Python is too old - uv is required
        echo ""
        echo "⚠️  Your system has Python $sys_python_ver (recipes require $MIN_PYTHON_VERSION)"
        echo ""
        show_uv_benefits
        echo ""
        read -p "Install 'uv $PINNED_UV_VERSION'? [Y/n]: " -n 1 -r
        echo

        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            if install_uv; then
                create_venv_with_uv
                return 0
            fi
            # install_uv already printed error message
            show_python_error
        else
            show_python_error
        fi
    fi
}

# Create venv with uv (can specify Python version)
create_venv_with_uv() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VENV_DIR="$(realpath "$SCRIPT_DIR/../llmb_venv")"

    echo "Creating virtual environment with uv..."
    if ! uv venv --clear -p "$RECOMMENDED_PYTHON_VERSION" "$VENV_DIR"; then
        echo "❌ Failed to create venv with uv"
        exit 1
    fi

    source "$VENV_DIR/bin/activate"
    uv pip install pip # Add pip for compatibility
    CREATED_VENV=true
    echo "✅ Virtual environment created and activated (Python $RECOMMENDED_PYTHON_VERSION)"
}

# Create venv with system python3
create_venv_with_python() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VENV_DIR="$(realpath "$SCRIPT_DIR/../llmb_venv")"

    echo "Creating virtual environment with system Python..."

    # Temporarily disable strict error handling for venv creation
    set +e
    python3 -m venv --clear "$VENV_DIR"
    venv_result=$?
    set -e

    if [ $venv_result -ne 0 ]; then
        echo "❌ Failed to create venv with python3 (exit code: $venv_result)"
        echo "DEBUG: Testing what went wrong..."
        echo "Python path: $(command -v python3)"
        echo "Python version: $(python3 --version)"
        echo "Target directory: $VENV_DIR"
        echo "Testing ensurepip: $(python3 -m ensurepip --help > /dev/null 2>&1 && echo 'OK' || echo 'FAILED')"
        exit 1
    fi

    source "$VENV_DIR/bin/activate"
    CREATED_VENV=true
    echo "✅ Virtual environment created and activated"
}
# Summarize environment and key next steps before launching llmb-install
print_preinstall_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "✅ Environment Ready"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""

    # Show venv info if we created one
    if [[ $CREATED_VENV == true ]]; then
        echo "📦 Virtual Environment Created"
        echo "   Location: $VENV_DIR"
        echo "   Python:   $(get_python_version python3)"
        echo ""
        echo "   To use this environment in future sessions:"
        echo "   source $VENV_DIR/bin/activate"
        echo ""
    fi

    # Show PATH info if we added uv to PATH
    if [[ $UV_INSTALLED_NOW == true && $UV_WAS_ADDED_TO_PATH == true ]]; then
        echo "⚡ uv Installed"
        echo "   Location: $HOME/.local/bin/uv"
        echo ""
        echo "⚠️  IMPORTANT: Add uv to your PATH permanently"
        echo "   Add this line to ~/.bashrc (or ~/.zshrc):"
        echo ""
        # shellcheck disable=SC2016 # Intentionally showing literal $PATH for user to copy
        echo "   export PATH=\"$HOME/.local/bin:\$PATH\""
        echo ""
        echo "   Then run: source ~/.bashrc"
        echo ""
        echo "   Note: uv is already available in this session."
        echo ""
    fi

    echo "────────────────────────────────────────────────────────────────────"
    echo "Press Enter to launch the main installer..."
    read -r
    echo ""
}
# Main execution: Setup environment and validate tools
setup_environment

# Determine which package manager to use for installation
USE_UV=false
if command_exists uv; then
    USE_UV=true
fi

# Helper function to install a package
install_package() {
    local package_name="$1"
    local package_dir="$2"

    pushd "$package_dir" > /dev/null
    if [ "$USE_UV" = true ]; then
        echo "  • Installing $package_name (with uv)..."
        uv pip install --quiet .
    else
        echo "  • Installing $package_name..."
        python3 -m pip install --quiet .
    fi
    popd > /dev/null
}

echo ""
echo "📦 Installing core tools..."

# Install runner and installer dependencies
install_package "llmb-run" "cli/llmb-run"
install_package "llmb-install" "cli/llmb-install"

echo "✅ Core tools installed successfully"

# Run the interactive installer (show summary only if we created a venv or installed uv)
if [[ $CREATED_VENV == true || $UV_INSTALLED_NOW == true || $UV_WAS_ADDED_TO_PATH == true ]]; then
    print_preinstall_summary
fi

echo ""
echo "🚀 Launching main installer..."
echo ""
llmb-install "$@"
