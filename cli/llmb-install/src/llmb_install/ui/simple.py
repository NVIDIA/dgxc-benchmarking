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


"""Simple print-based UI implementation for LLMB Install."""

from typing import Any, Dict, List, Optional

import questionary

from llmb_install.ui.interface import UIInterface


class SimpleUI(UIInterface):
    """Simple print-based UI implementation using questionary."""

    def prompt_select(
        self, message: str, choices: List[Dict[str, Any]], default: Optional[str] = None
    ) -> Optional[str]:
        """Prompt user to select from choices using questionary."""
        if not choices:
            return None

        # Convert to questionary format if needed
        formatted_choices = []
        default_choice = None

        for choice in choices:
            if isinstance(choice, dict):
                formatted_choices.append(choice)
                if default and choice.get('value') == default:
                    default_choice = choice
            else:
                # Simple string choice
                formatted_choices.append({'name': str(choice), 'value': str(choice)})
                if default and str(choice) == default:
                    default_choice = formatted_choices[-1]

        return questionary.select(message, choices=formatted_choices, default=default_choice).ask()

    def prompt_text(self, message: str, default: str = "", validate: Optional[callable] = None) -> Optional[str]:
        """Prompt user for text input using questionary."""
        return questionary.text(message, default=default, validate=validate).ask()

    def prompt_confirm(self, message: str, default: bool = True) -> Optional[bool]:
        """Prompt user for yes/no confirmation using questionary."""
        return questionary.confirm(message, default=default).ask()

    def display_progress(self, message: str, current: int, total: int) -> None:
        """Display simple progress information."""
        percentage = int((current / total) * 100) if total > 0 else 0
        print(f"{message} [{current}/{total}] ({percentage}%)")

    def log(self, message: str, level: str = 'info') -> None:
        """Print message with simple level formatting."""
        prefix_map = {'info': '', 'warning': 'WARNING: ', 'error': 'ERROR: ', 'success': 'SUCCESS: '}
        prefix = prefix_map.get(level, '')
        print(f"{prefix}{message}")

    def print_section(self, title: str, content: Optional[str] = None) -> None:
        """Print section with simple formatting."""
        print(f"\n{title}")
        print("-" * len(title))
        if content:
            print(content)
        print()

    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Print simple table format."""
        if not headers and not rows:
            return

        # Calculate column widths
        all_rows = [headers] + rows if headers else rows
        if not all_rows:
            return

        col_widths = []
        max_cols = max(len(row) for row in all_rows)

        for col in range(max_cols):
            max_width = 0
            for row in all_rows:
                if col < len(row):
                    max_width = max(max_width, len(str(row[col])))
            col_widths.append(max_width)

        def print_row(row: List[str], separator: str = " | "):
            formatted_cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    formatted_cells.append(str(cell).ljust(col_widths[i]))
                else:
                    formatted_cells.append(str(cell))
            print(separator.join(formatted_cells))

        # Print headers if provided
        if headers:
            print_row(headers)
            print_row(["-" * width for width in col_widths])

        # Print data rows
        for row in rows:
            print_row(row)

    def prompt_path(self, message: str, default: str = "", validate: Optional[callable] = None) -> Optional[str]:
        """Prompt the user for a file/directory path.

        Args:
            message: The prompt message
            default: Default path value
            validate: Optional validation function that returns 'valid', 'invalid:message', etc.

        Returns:
            The selected path or None if cancelled
        """

        def questionary_validate(path: str) -> bool | str:
            if validate:
                result = validate(path)
                if result == "valid":
                    return True
                elif result.startswith("invalid:"):
                    return result.split(":", 1)[1]
                else:
                    return True
            return True

        return questionary.path(message, default=default, validate=questionary_validate).ask()

    def prompt_checkbox(
        self,
        message: str,
        choices: List[Dict[str, str]],
        validate: Optional[callable] = None,
        default: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Prompt the user to select multiple items from a list.

        Args:
            message: The prompt message
            choices: List of choice dictionaries with 'name' and 'value' keys
            validate: Optional validation function that returns 'valid', 'invalid:message', etc.
            default: List of values to pre-select

        Returns:
            List of selected values or None if cancelled
        """

        def questionary_validate(selected: List[str]) -> bool | str:
            if validate:
                result = validate(selected)
                if result == "valid":
                    return True
                elif result.startswith("invalid:"):
                    return result.split(":", 1)[1]
                else:
                    return True
            return True

        # Convert choices to questionary Choice objects with pre-selection
        questionary_choices = []
        for choice in choices:
            checked = default and choice['value'] in default
            questionary_choices.append(questionary.Choice(choice['name'], value=choice['value'], checked=checked))

        return questionary.checkbox(message, choices=questionary_choices, validate=questionary_validate).ask()
