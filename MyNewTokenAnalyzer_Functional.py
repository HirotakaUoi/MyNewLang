#!/usr/bin/python3
# coding=utf-8

from dataclasses import dataclass
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclass(frozen=True)
class Position:
    index: int
    line: int
    column: int


PositionTuple = Tuple[int, int, int]
TokenTuple = Tuple[str, Any, PositionTuple, PositionTuple]
TokenWithExtras = Tuple[str, Any, PositionTuple, PositionTuple, Dict[str, Any]]


@dataclass(frozen=True)
class Token:
    kind: str
    value: Any
    start: Position
    end: Position  # end is exclusive
    extras: Optional[Dict[str, Any]] = None

    def as_tuple(self) -> Union[TokenTuple, TokenWithExtras]:
        base = (self.kind, self.value, _pos_to_tuple(self.start), _pos_to_tuple(self.end))
        if self.extras:
            return base + (self.extras,)
        return base


class TokenizeError(Exception):
    def __init__(self, message, position):
        super().__init__(message)
        self.position = _coerce_pos(position)


def _pos_to_tuple(pos: Position) -> PositionTuple:
    return (pos.index, pos.line, pos.column)


def _coerce_pos(pos: Union[Position, PositionTuple]) -> PositionTuple:
    if isinstance(pos, Position):
        return _pos_to_tuple(pos)
    return pos


def build_standard_operators():
    return [
        "==", "!=", ">=", "<=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=",
        "++", "--",
        "<-",
        "=",
        "+", "-", "*", "/", "%", "^", "<", ">"
    ]


def build_standard_pairs():
    return [
        ("(", ")"),
        ("{", "}"),
        ("[", "]"),
    ]


def build_standard_comments():
    return ["//"]


def build_standard_keywords():
    return [
        "break", "continue", "do", "else", "for", "if", "return", "while",
    ]


def _strip_comment(line):
    in_quote = False
    for idx, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "%" and not in_quote:
            return line[:idx].strip()
    return line.strip()


def parse_definition_lines(lines):
    operators = []
    pairs = []
    comments = []
    keywords = []

    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue
        if not line.endswith("."):
            continue
        body = line[:-1].strip()

        if body.startswith("op(") and body.endswith(")"):
            inner = body[3:-1].strip()
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) != 3:
                continue
            lexeme = parts[0]
            if lexeme.startswith("'") and lexeme.endswith("'") and len(lexeme) >= 2:
                lexeme = lexeme[1:-1]
            operators.append(lexeme)
            continue

        if body.startswith("pair(") and body.endswith(")"):
            inner = body[5:-1].strip()
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) != 2:
                continue
            left, right = parts
            if left.startswith("'") and left.endswith("'") and len(left) >= 2:
                left = left[1:-1]
            if right.startswith("'") and right.endswith("'") and len(right) >= 2:
                right = right[1:-1]
            pairs.append((left, right))
            continue

        if body.startswith("comment(") and body.endswith(")"):
            inner = body[8:-1].strip()
            if inner.startswith("'") and inner.endswith("'") and len(inner) >= 2:
                inner = inner[1:-1]
            comments.append(inner)
            continue

        if body.startswith("keyword(") and body.endswith(")"):
            inner = body[8:-1].strip()
            if inner.startswith("'") and inner.endswith("'") and len(inner) >= 2:
                inner = inner[1:-1]
            keywords.append(inner)
            continue

    return operators, pairs, comments, keywords


def load_definitions_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return parse_definition_lines(f.readlines())


def build_definition_set(source_path=None, module_extension=".mydef"):
    operators = list(build_standard_operators())
    pairs = list(build_standard_pairs())
    comments = list(build_standard_comments())
    keywords = list(build_standard_keywords())

    if source_path:
        import os
        base_dir = os.path.dirname(source_path)
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        module_path = os.path.join(base_dir, base_name + module_extension)
        if os.path.isfile(module_path):
            mod_ops, mod_pairs, mod_comments, mod_keywords = load_definitions_from_file(module_path)
            operators.extend(mod_ops)
            pairs.extend(mod_pairs)
            comments.extend(mod_comments)
            keywords.extend(mod_keywords)

    return operators, pairs, comments, keywords


@dataclass(frozen=True)
class State:
    source: str
    index: int
    line: int
    column: int
    offset: int


Parser = Callable[[State], Optional[Tuple[Any, State]]]


def _current_pos(state: State) -> Position:
    return Position(state.offset + state.index, state.line, state.column)


def _advance(state: State, n: int = 1) -> State:
    idx = state.index
    line = state.line
    col = state.column
    src = state.source
    length = len(src)
    for _ in range(n):
        if idx >= length:
            break
        if src[idx] == "\n":
            line += 1
            col = 1
        else:
            col += 1
        idx += 1
    return State(src, idx, line, col, state.offset)


def _take_while(state: State, pred: Callable[[str], bool]) -> Tuple[str, State]:
    src = state.source
    cur = state
    while cur.index < len(src) and pred(src[cur.index]):
        cur = _advance(cur, 1)
    return src[state.index:cur.index], cur


def _p_map(parser: Parser, fn: Callable[[Any], Any]) -> Parser:
    def run(state: State):
        res = parser(state)
        if res is None:
            return None
        value, next_state = res
        return fn(value), next_state
    return run


def _p_or(*parsers: Parser) -> Parser:
    def run(state: State):
        for parser in parsers:
            res = parser(state)
            if res is not None:
                return res
        return None
    return run


def _is_alpha(ch: str) -> bool:
    return ch.isalpha()


def _is_digit(ch: str) -> bool:
    return ch.isdigit()


def _is_alnum(ch: str) -> bool:
    return ch.isalnum()


def _is_unicode_symbol(ch: str) -> bool:
    return unicodedata.category(ch) == "So"


def _is_ident_start(ch: str) -> bool:
    return _is_alpha(ch) or ch == "_" or _is_unicode_symbol(ch)


def _is_ident_char(ch: str) -> bool:
    return _is_alnum(ch) or ch == "_" or _is_unicode_symbol(ch)


def _is_dollar_ident_char(ch: str) -> bool:
    return _is_alnum(ch) or _is_unicode_symbol(ch)


def _is_word_lexeme(lexeme: str) -> bool:
    return bool(lexeme) and _is_ident_start(lexeme[0])


def _check_word_boundary(state: State, lexeme: str) -> bool:
    if not _is_word_lexeme(lexeme):
        return True
    next_idx = state.index + len(lexeme)
    if next_idx >= len(state.source):
        return True
    return not _is_ident_char(state.source[next_idx])


class FunctionalTokenizer:
    def __init__(self, operators=None, pairs=None, comments=None, keywords=None):
        if operators is None:
            operators = build_standard_operators()
        if pairs is None:
            pairs = build_standard_pairs()
        if comments is None:
            comments = build_standard_comments()
        if keywords is None:
            keywords = build_standard_keywords()
        self._operator_set = set(operators)
        self.operators = sorted(self._operator_set, key=len, reverse=True)
        self._comment_set = set(comments)
        self.comment_starters = sorted(self._comment_set, key=len, reverse=True)
        self._pair_list = list(pairs)
        self.pairs = list(self._pair_list)
        self._keyword_set = set(keywords)
        self.keywords = sorted(self._keyword_set, key=len, reverse=True)
        self._operator_stack = []
        self._comment_stack = []
        self._pair_stack = []
        self._keyword_stack = []
        self._refresh_pair_lexemes()

    def _refresh_operator_list(self):
        self.operators = sorted(self._operator_set, key=len, reverse=True)

    def _refresh_comment_list(self):
        self.comment_starters = sorted(self._comment_set, key=len, reverse=True)

    def _refresh_keyword_list(self):
        self.keywords = sorted(self._keyword_set, key=len, reverse=True)

    def _refresh_pair_lexemes(self):
        lexemes = set()
        for left, right in self._pair_list:
            lexemes.add(left)
            lexemes.add(right)
        self._pair_lexemes = sorted(lexemes, key=len, reverse=True)

    def push_operator_update(self, add=None, remove=None):
        self._operator_stack.append(set(self._operator_set))
        if add:
            self._operator_set.update(add)
        if remove:
            self._operator_set.difference_update(remove)
        self._refresh_operator_list()

    def pop_operator_update(self):
        if not self._operator_stack:
            return False
        self._operator_set = self._operator_stack.pop()
        self._refresh_operator_list()
        return True

    def push_comment_update(self, add=None, remove=None):
        self._comment_stack.append(set(self._comment_set))
        if add:
            self._comment_set.update(add)
        if remove:
            self._comment_set.difference_update(remove)
        self._refresh_comment_list()

    def pop_comment_update(self):
        if not self._comment_stack:
            return False
        self._comment_set = self._comment_stack.pop()
        self._refresh_comment_list()
        return True

    def push_keyword_update(self, add=None, remove=None):
        self._keyword_stack.append(set(self._keyword_set))
        if add:
            self._keyword_set.update(add)
        if remove:
            self._keyword_set.difference_update(remove)
        self._refresh_keyword_list()

    def pop_keyword_update(self):
        if not self._keyword_stack:
            return False
        self._keyword_set = self._keyword_stack.pop()
        self._refresh_keyword_list()
        return True

    def push_pair_update(self, add=None, remove=None):
        self._pair_stack.append(list(self._pair_list))
        if add:
            self._pair_list.extend(add)
        if remove:
            self._pair_list = [p for p in self._pair_list if p not in remove]
        self.pairs = list(self._pair_list)
        self._refresh_pair_lexemes()

    def pop_pair_update(self):
        if not self._pair_stack:
            return False
        self._pair_list = self._pair_stack.pop()
        self.pairs = list(self._pair_list)
        self._refresh_pair_lexemes()
        return True

    def _classify_identifier(self, value: str, start_pos: Position) -> str:
        if not self.keywords:
            return "IDENT"
        for kw in self.keywords:
            if value == kw:
                return "KEYWORD"
            if value.startswith(kw):
                raise TokenizeError("Identifier starts with reserved keyword", start_pos)
        return "IDENT"

    def _comment_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            matched = None
            for starter in self.comment_starters:
                if src.startswith(starter, state.index):
                    matched = starter
                    break
            if matched is None:
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, len(matched))
            content_start = cur.index
            _, cur = _take_while(cur, lambda ch: ch != "\n")
            value = matched + src[content_start:cur.index]
            return Token("COMMENT", value, start_pos, _current_pos(cur)), cur
        return run

    def _string_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            if state.index >= len(src) or src[state.index] != '"':
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, 1)
            content_start = cur.index
            while cur.index < len(src) and src[cur.index] != '"':
                cur = _advance(cur, 1)
            if cur.index >= len(src):
                raise TokenizeError("Unterminated string literal", start_pos)
            content = src[content_start:cur.index]
            cur = _advance(cur, 1)
            return Token("STRING", content, start_pos, _current_pos(cur)), cur
        return run

    def _number_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            if state.index >= len(src) or not _is_digit(src[state.index]):
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, 1)
            _, cur = _take_while(cur, _is_digit)
            mantissa = src[start_pos.index:cur.index]
            value = int(mantissa)
            if cur.index + 1 < len(src) and src[cur.index] == "E" and _is_digit(src[cur.index + 1]):
                cur = _advance(cur, 1)
                exp_start = cur.index
                _, cur = _take_while(cur, _is_digit)
                exponent = int(src[exp_start:cur.index])
                value = value * (10 ** exponent)
            return Token("NUMBER", value, start_pos, _current_pos(cur)), cur
        return run

    def _quoted_ident_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            if state.index >= len(src) or src[state.index] != "'":
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, 1)
            content_start = cur.index
            while cur.index < len(src) and src[cur.index] != "'":
                cur = _advance(cur, 1)
            if cur.index >= len(src):
                raise TokenizeError("Unterminated quoted identifier", start_pos)
            content = src[content_start:cur.index]
            cur = _advance(cur, 1)
            value = "'" + content + "'"
            kind = self._classify_identifier(value, start_pos)
            return Token(kind, value, start_pos, _current_pos(cur)), cur
        return run

    def _ident_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            if state.index >= len(src) or not _is_ident_start(src[state.index]):
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, 1)
            _, cur = _take_while(cur, _is_ident_char)
            value = src[start_pos.index:cur.index]
            kind = self._classify_identifier(value, start_pos)
            return Token(kind, value, start_pos, _current_pos(cur)), cur
        return run

    def _dollar_ident_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            if state.index >= len(src) or src[state.index] != "$":
                return None
            if state.index + 1 >= len(src) or not _is_dollar_ident_char(src[state.index + 1]):
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, 1)
            _, cur = _take_while(cur, _is_dollar_ident_char)
            value = src[start_pos.index:cur.index]
            kind = self._classify_identifier(value, start_pos)
            return Token(kind, value, start_pos, _current_pos(cur)), cur
        return run

    def _operator_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            matched = None
            for op in self.operators:
                if src.startswith(op, state.index):
                    matched = op
                    break
            start_pos = _current_pos(state)
            if matched is None:
                cur = _advance(state, 1)
                value = src[start_pos.index:cur.index]
                return Token("PUNC", value, start_pos, _current_pos(cur)), cur
            cur = _advance(state, len(matched))
            return Token("OP", matched, start_pos, _current_pos(cur)), cur
        return run

    def _pair_parser(self) -> Parser:
        def run(state: State):
            src = state.source
            matched = None
            for lexeme in self._pair_lexemes:
                if src.startswith(lexeme, state.index) and _check_word_boundary(state, lexeme):
                    matched = lexeme
                    break
            if matched is None:
                return None
            start_pos = _current_pos(state)
            cur = _advance(state, len(matched))
            return Token("OP", matched, start_pos, _current_pos(cur)), cur
        return run

    def tokenize(self, source: str, start_pos: PositionTuple = (0, 1, 1)) -> List[Tuple]:
        tokens: List[Token] = []
        base_index, base_line, base_col = start_pos
        state = State(source, 0, base_line, base_col, base_index)
        length = len(source)

        parse_token = _p_or(
            self._string_parser(),
            self._comment_parser(),
            self._number_parser(),
            self._quoted_ident_parser(),
            self._ident_parser(),
            self._dollar_ident_parser(),
            self._pair_parser(),
            self._operator_parser(),
        )

        while state.index < length:
            ch = source[state.index]
            if ch.isspace():
                state = _advance(state, 1)
                continue

            res = parse_token(state)
            if res is None:
                start_pos = _current_pos(state)
                raise TokenizeError("Unexpected character", start_pos)
            token, state = res
            tokens.append(token)

        return [t.as_tuple() for t in tokens]


def tokenize(
    source,
    operators=None,
    pairs=None,
    comments=None,
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    return FunctionalTokenizer(operators, pairs, comments, keywords).tokenize(source, start_pos)


def tokenize_with_definitions(
    source,
    source_path,
    module_extension=".mydef",
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    operators, pairs, comments, def_keywords = build_definition_set(source_path, module_extension)
    if keywords:
        def_keywords = list(def_keywords) + list(keywords)
    return FunctionalTokenizer(operators, pairs, comments, def_keywords).tokenize(source, start_pos)


def tokenize_lines(
    lines,
    source_path=None,
    module_extension=".mydef",
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    source = "".join(lines)
    if source_path:
        return tokenize_with_definitions(
            source,
            source_path=source_path,
            module_extension=module_extension,
            keywords=keywords,
            start_pos=start_pos,
        )
    return tokenize(source, keywords=keywords, start_pos=start_pos)


if __name__ == "__main__":
    sample = "var $x = foo + 'bar' // comment\nx <- y"
    for t in tokenize(sample, start_pos=(5, 5, 1)):
        print(t)
