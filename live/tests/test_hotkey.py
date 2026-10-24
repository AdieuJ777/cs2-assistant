"""Tests for hotkey argument handling that do not need pynput installed."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from live.midround.hotkey import _resolve_key, main  # noqa: E402


class FakeKeyboard:
    class Key:
        f8 = "KEY_F8"
        esc = "KEY_ESC"

    class KeyCode:
        @staticmethod
        def from_char(char: str) -> str:
            return f"CHAR:{char}"


class HotkeyTests(unittest.TestCase):
    def test_resolve_function_key(self) -> None:
        self.assertEqual(_resolve_key("f8", FakeKeyboard), "KEY_F8")
        self.assertEqual(_resolve_key("F8", FakeKeyboard), "KEY_F8")

    def test_resolve_character(self) -> None:
        self.assertEqual(_resolve_key("x", FakeKeyboard), "CHAR:x")

    def test_resolve_unknown(self) -> None:
        self.assertIsNone(_resolve_key("not-a-key", FakeKeyboard))

    def test_key_and_button_are_mutually_exclusive(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--key", "f8", "--button", "x1"])


if __name__ == "__main__":
    unittest.main()
