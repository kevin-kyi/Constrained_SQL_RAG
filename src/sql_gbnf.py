from __future__ import annotations

from pathlib import Path
from typing import Iterable

from gbnf import GBNF
from gbnf.grammar_graph.grammar_graph_types import RuleChar, RuleCharExclude, RuleEnd
from gbnf.grammar_graph.type_guards import (
    is_range,
    is_rule_char,
    is_rule_char_exclude,
    is_rule_end,
    is_rule_ref,
)
from lmformatenforcer import CharacterLevelParser, CharacterLevelParserConfig
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

DEFAULT_SQLITE_GBNF = Path(__file__).resolve().parent / "sql_grammar/sqlite.gbnf"


def load_sqlite_grammar(path: str | Path = DEFAULT_SQLITE_GBNF) -> str:
    """Load the SQLite-oriented SQL grammar as a string."""
    grammar_path = Path(path)
    return grammar_path.read_text(encoding="utf-8")


class SQLGBNFParser(CharacterLevelParser):
    """Character-level parser backed by a GBNF grammar graph."""

    def __init__(
        self,
        grammar: str,
        config: CharacterLevelParserConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._grammar = grammar
        self._state = GBNF(grammar)
        self._alphabet_codepoints = [ord(ch) for ch in self.config.alphabet]

    @classmethod
    def _from_state(
        cls,
        grammar: str,
        state,
        config: CharacterLevelParserConfig,
        alphabet_codepoints: list[int],
    ) -> "SQLGBNFParser":
        parser = cls.__new__(cls)
        CharacterLevelParser.__init__(parser, config)
        parser._grammar = grammar
        parser._state = state
        parser._alphabet_codepoints = alphabet_codepoints
        return parser

    @property  # type: ignore[override]
    def config(self) -> CharacterLevelParserConfig:
        return self._config

    @config.setter  # type: ignore[override]
    def config(self, new_config: CharacterLevelParserConfig) -> None:
        self._config = new_config
        self._alphabet_codepoints = [ord(ch) for ch in new_config.alphabet]

    def add_character(self, new_character: str) -> "SQLGBNFParser":
        if len(new_character) != 1:
            raise ValueError("GBNF parser expects single characters at a time")
        next_state = self._state.add(new_character)
        return self._from_state(
            self._grammar,
            next_state,
            self.config,
            self._alphabet_codepoints,
        )

    def _iter_rules(self):
        pointers = getattr(self._state, "__pointers__", [])
        for pointer in pointers:
            yield pointer.rule

    def get_allowed_characters(self) -> str:
        allowed: set[int] = set()

        for rule in self._iter_rules():
            if is_rule_ref(rule):
                continue
            if is_rule_char(rule):
                allowed.update(self._expand_values(rule.value))
            elif is_rule_char_exclude(rule):
                allowed.update(self._complement_values(rule.value))

        ordered = sorted(allowed)
        return "".join(chr(cp) for cp in ordered)

    def can_end(self) -> bool:
        return any(is_rule_end(rule) for rule in self._iter_rules())

    def shortcut_key(self):
        pointers = getattr(self._state, "__pointers__", [])
        ids = tuple(pointer.node.id for pointer in pointers)
        return ids

    def cache_key(self):
        return self.shortcut_key()

    def _expand_values(self, values: Iterable[int | list[int]]) -> set[int]:
        expanded: set[int] = set()
        for value in values:
            if is_range(value):
                expanded.update(range(value[0], value[1] + 1))
            else:
                expanded.add(value)
        return expanded

    def _complement_values(self, values: Iterable[int | list[int]]) -> set[int]:
        excluded = self._expand_values(values)
        return {cp for cp in self._alphabet_codepoints if cp not in excluded}


def build_sqlite_prefix_allowed_tokens_fn(
    tokenizer,
    grammar_text: str | None = None,
) -> callable:
    """Build a prefix_allowed_tokens_fn that constrains SQL generation to the SQLite grammar."""
    grammar = grammar_text or load_sqlite_grammar()
    parser = SQLGBNFParser(grammar)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
