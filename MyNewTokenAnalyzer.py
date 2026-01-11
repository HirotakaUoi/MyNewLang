#!/usr/bin/python3
# coding=utf-8

from dataclasses import dataclass
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union


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
    end: Position  # endは排他的（この位置の文字は含まない）
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
    """Position をタプルに変換する。"""
    return (pos.index, pos.line, pos.column)


def _coerce_pos(pos: Union[Position, PositionTuple]) -> PositionTuple:
    """Position またはタプルを (index, line, column) 形式に揃える。"""
    if isinstance(pos, Position):
        return _pos_to_tuple(pos)
    return pos


def build_standard_operators():
    """標準の演算子定義を返す。"""
    # 標準の演算子定義（長いものほど優先される）
    return [
        "==", "!=", ">=", "<=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=",
        "++", "--",
        "<-",
        "=",
        "+", "-", "*", "/", "%", "^", "<", ">"
    ]


def build_standard_pairs():
    """標準の括弧対を返す。"""
    # 括弧対（字句解析では OP として出力）
    return [
        ("(", ")"),
        ("{", "}"),
        ("[", "]"),
    ]


def build_standard_comments():
    """標準のコメント開始記号を返す。"""
    # 行コメントの開始記号
    return ["//"]


def build_standard_keywords():
    """標準キーワード一覧を返す。"""
    # キーワード（前方一致はエラー）
    return [
        "break", "continue", "do", "else", "for", "if", "return", "while",
    ]


def _strip_comment(line):
    """mydef の行から % コメントを除去する。"""
    # % コメントを除去する（ただしシングルクォート内の % は無視する）
    in_quote = False
    for idx, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "%" and not in_quote:
            return line[:idx].strip()
    return line.strip()


def parse_definition_lines(lines):
    """定義ファイルの行から演算子/括弧対/コメント/キーワードを抽出する。"""
    # .mydef から演算子/括弧対/コメント/キーワードを抽出する
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

        # op('<lexeme>', <precedence>, <assoc>).
        # 字句解析では lexeme のみを使う（他は構文解析向け）
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

        # pair('<open>', '<close>').
        # begin...end のような括弧対の宣言用
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

        # comment('<prefix>').
        # コメント開始トークン（例: //）の宣言用
        if body.startswith("comment(") and body.endswith(")"):
            inner = body[8:-1].strip()
            if inner.startswith("'") and inner.endswith("'") and len(inner) >= 2:
                inner = inner[1:-1]
            comments.append(inner)
            continue

        # keyword('<name>').
        if body.startswith("keyword(") and body.endswith(")"):
            inner = body[8:-1].strip()
            if inner.startswith("'") and inner.endswith("'") and len(inner) >= 2:
                inner = inner[1:-1]
            keywords.append(inner)
            continue

    return operators, pairs, comments, keywords


def load_definitions_from_file(path):
    """定義ファイルを読み込み、定義一覧を返す。"""
    with open(path, "r", encoding="utf-8") as f:
        return parse_definition_lines(f.readlines())


def build_definition_set(source_path=None, module_extension=".mydef"):
    """標準定義と同名 .mydef を統合した定義セットを返す。"""
    # 標準定義に加えて、ソースと同名の .mydef 定義を読み込んで拡張する
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


class Tokenizer:
    def __init__(self, operators=None, pairs=None, comments=None, keywords=None):
        """トークナイザを初期化する。"""
        if operators is None:
            operators = build_standard_operators()
        if pairs is None:
            pairs = build_standard_pairs()
        if comments is None:
            comments = build_standard_comments()
        if keywords is None:
            keywords = build_standard_keywords()
        # 複数文字の演算子・コメント開始記号は最長一致にする
        self._operator_set = set(operators)
        self.operators = sorted(self._operator_set, key=len, reverse=True)
        self._comment_set = set(comments)
        self.comment_starters = sorted(self._comment_set, key=len, reverse=True)
        self._pair_list = list(pairs)
        self.pairs = list(self._pair_list)
        self._keyword_set = set(keywords)
        self.keywords = sorted(self._keyword_set, key=len, reverse=True)
        # 更新を取り消すための履歴
        self._operator_stack = []
        self._comment_stack = []
        self._pair_stack = []
        self._keyword_stack = []
        self._refresh_pair_lexemes()

    def _refresh_operator_list(self):
        """演算子の最長一致用リストを再構築する。"""
        # 演算子集合から最長一致用の並びを再構築する
        self.operators = sorted(self._operator_set, key=len, reverse=True)

    def _refresh_comment_list(self):
        """コメント開始記号の最長一致用リストを再構築する。"""
        # コメント開始記号の最長一致用の並びを再構築する
        self.comment_starters = sorted(self._comment_set, key=len, reverse=True)

    def _refresh_keyword_list(self):
        """キーワード一覧を再構築する。"""
        self.keywords = sorted(self._keyword_set, key=len, reverse=True)

    def _refresh_pair_lexemes(self):
        """括弧対の字面一覧を再構築する。"""
        lexemes = set()
        for left, right in self._pair_list:
            lexemes.add(left)
            lexemes.add(right)
        self._pair_lexemes = sorted(lexemes, key=len, reverse=True)

    def push_operator_update(self, add=None, remove=None):
        """演算子定義を更新し、戻せるように退避する。"""
        # 現在の演算子集合を退避し、更新を適用する
        self._operator_stack.append(set(self._operator_set))
        if add:
            self._operator_set.update(add)
        if remove:
            self._operator_set.difference_update(remove)
        self._refresh_operator_list()

    def pop_operator_update(self):
        """直前の演算子更新を取り消す。"""
        # 直前の演算子更新を取り消す
        if not self._operator_stack:
            return False
        self._operator_set = self._operator_stack.pop()
        self._refresh_operator_list()
        return True

    def push_comment_update(self, add=None, remove=None):
        """コメント開始記号を更新し、戻せるように退避する。"""
        # 現在のコメント開始記号を退避し、更新を適用する
        self._comment_stack.append(set(self._comment_set))
        if add:
            self._comment_set.update(add)
        if remove:
            self._comment_set.difference_update(remove)
        self._refresh_comment_list()

    def pop_comment_update(self):
        """直前のコメント更新を取り消す。"""
        # 直前のコメント更新を取り消す
        if not self._comment_stack:
            return False
        self._comment_set = self._comment_stack.pop()
        self._refresh_comment_list()
        return True

    def push_keyword_update(self, add=None, remove=None):
        """キーワード定義を更新し、戻せるように退避する。"""
        self._keyword_stack.append(set(self._keyword_set))
        if add:
            self._keyword_set.update(add)
        if remove:
            self._keyword_set.difference_update(remove)
        self._refresh_keyword_list()

    def pop_keyword_update(self):
        """直前のキーワード更新を取り消す。"""
        if not self._keyword_stack:
            return False
        self._keyword_set = self._keyword_stack.pop()
        self._refresh_keyword_list()
        return True

    def push_pair_update(self, add=None, remove=None):
        """括弧対定義を更新し、戻せるように退避する。"""
        # 現在の括弧対を退避し、更新を適用する
        self._pair_stack.append(list(self._pair_list))
        if add:
            self._pair_list.extend(add)
        if remove:
            self._pair_list = [p for p in self._pair_list if p not in remove]
        self.pairs = list(self._pair_list)
        self._refresh_pair_lexemes()

    def pop_pair_update(self):
        """直前の括弧対更新を取り消す。"""
        # 直前の括弧対更新を取り消す
        if not self._pair_stack:
            return False
        self._pair_list = self._pair_stack.pop()
        self.pairs = list(self._pair_list)
        self._refresh_pair_lexemes()
        return True

    def tokenize(self, source, start_pos: PositionTuple = (0, 1, 1)) -> List[Tuple]:
        """入力文字列をトークン列に変換する。"""
        # 文字列を走査してトークン列に変換する
        tokens: List[Token] = []
        i = 0
        base_index, line, col = start_pos
        length = len(source)

        def current_pos():
            # 入力全体での位置（index/line/column）を返す
            return Position(base_index + i, line, col)

        def advance(n=1):
            # 文字位置を進める。行・列も同時に更新する
            nonlocal i, line, col
            for _ in range(n):
                if i >= length:
                    return
                if source[i] == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1

        def emit(kind, value, start_pos, end_pos, extras: Optional[Dict[str, Any]] = None):
            # トークンを追加する
            tokens.append(Token(kind, value, start_pos, end_pos, extras))

        def is_alpha(ch):
            return ch.isalpha()

        def is_digit(ch):
            return ch.isdigit()

        def is_alnum(ch):
            return ch.isalnum()

        def is_unicode_symbol(ch):
            return unicodedata.category(ch) == "So"

        def is_ident_start(ch):
            return is_alpha(ch) or ch == "_" or is_unicode_symbol(ch)

        def is_ident_char(ch):
            return is_alnum(ch) or ch == "_" or is_unicode_symbol(ch)

        def is_dollar_ident_char(ch):
            return is_alnum(ch) or is_unicode_symbol(ch)

        def is_word_lexeme(lexeme):
            if not lexeme:
                return False
            return is_ident_start(lexeme[0])

        def check_word_boundary(lexeme):
            if not is_word_lexeme(lexeme):
                return True
            next_idx = i + len(lexeme)
            if next_idx >= length:
                return True
            return not is_ident_char(source[next_idx])

        def classify_identifier(value, start_pos):
            if not self.keywords:
                return "IDENT"
            for kw in self.keywords:
                if value == kw:
                    return "KEYWORD"
                if value.startswith(kw):
                    raise TokenizeError("Identifier starts with reserved keyword", start_pos)
            return "IDENT"

        while i < length:
            ch = source[i]

            # Skip whitespace
            if ch.isspace():
                advance(1)
                continue

            # 文字列リテラル: "..."
            if ch == '"':
                start = current_pos()
                advance(1)
                content_start = i
                while i < length and source[i] != '"':
                    advance(1)
                if i >= length:
                    raise TokenizeError("Unterminated string literal", start)
                content = source[content_start:i]
                advance(1)  # closing quote
                end = current_pos()
                emit("STRING", content, start, end)
                continue

            # 行コメント: 定義済みの開始記号で始まる（改行までは含めない）
            matched_comment = None
            for starter in self.comment_starters:
                if source.startswith(starter, i):
                    matched_comment = starter
                    break
            if matched_comment is not None:
                start = current_pos()
                advance(len(matched_comment))
                comment_start = i
                while i < length and source[i] != "\n":
                    advance(1)
                value = matched_comment + source[comment_start:i]
                end = current_pos()
                emit("COMMENT", value, start, end)
                continue

            # 括弧対: 定義済みの開閉字面は OP として出力する
            matched_pair = None
            for lexeme in self._pair_lexemes:
                if source.startswith(lexeme, i) and check_word_boundary(lexeme):
                    matched_pair = lexeme
                    break
            if matched_pair is not None:
                start = current_pos()
                advance(len(matched_pair))
                end = current_pos()
                emit("OP", matched_pair, start, end)
                continue

            # 数値リテラル: [0-9]+ または [0-9]+E[0-9]+
            if is_digit(ch):
                start_i = i
                start = current_pos()
                advance(1)
                while i < length and is_digit(source[i]):
                    advance(1)
                mantissa = source[start_i:i]
                value = int(mantissa)
                if i + 1 < length and source[i] == "E" and is_digit(source[i + 1]):
                    advance(1)  # consume E
                    exp_start = i
                    while i < length and is_digit(source[i]):
                        advance(1)
                    exponent = int(source[exp_start:i])
                    value = value * (10 ** exponent)
                end = current_pos()
                emit("NUMBER", value, start, end)
                continue

            # クォート識別子: '...'
            if ch == "'":
                start = current_pos()
                advance(1)
                content_start = i
                while i < length and source[i] != "'":
                    advance(1)
                if i >= length:
                    raise TokenizeError("Unterminated quoted identifier", start)
                content = source[content_start:i]
                advance(1)  # closing quote
                end = current_pos()
                value = "'" + content + "'"
                kind = classify_identifier(value, start)
                emit(kind, value, start, end)
                continue

            # 標準識別子: [A-Za-z_][A-Za-z0-9_]*
            if is_ident_start(ch):
                start_i = i
                start = current_pos()
                advance(1)
                while i < length and is_ident_char(source[i]):
                    advance(1)
                end = current_pos()
                value = source[start_i:i]
                kind = classify_identifier(value, start)
                emit(kind, value, start, end)
                continue

            # $ から始まる識別子: $[A-Za-z0-9]+
            if ch == "$":
                if i + 1 < length and is_dollar_ident_char(source[i + 1]):
                    start_i = i
                    start = current_pos()
                    advance(1)
                    while i < length and is_dollar_ident_char(source[i]):
                        advance(1)
                    end = current_pos()
                    value = source[start_i:i]
                    kind = classify_identifier(value, start)
                    emit(kind, value, start, end)
                    continue

            # 演算子: 宣言済みを最長一致、それ以外は1文字
            matched = None
            for op in self.operators:
                if source.startswith(op, i):
                    matched = op
                    break
            start = current_pos()
            if matched is None:
                start_i = i
                advance(1)
                end = current_pos()
                emit("PUNC", source[start_i:i], start, end)
            else:
                advance(len(matched))
                end = current_pos()
                emit("OP", matched, start, end)

        return [t.as_tuple() for t in tokens]


def tokenize(
    source,
    operators=None,
    pairs=None,
    comments=None,
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    """入力文字列をトークン列に変換する（簡易API）。"""
    return Tokenizer(operators, pairs, comments, keywords).tokenize(source, start_pos)


def tokenize_with_definitions(
    source,
    source_path,
    module_extension=".mydef",
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    """同名 .mydef を考慮して入力文字列をトークン化する。"""
    # ソースファイル名に基づいて標準+ローカル定義を読み込む
    operators, pairs, comments, def_keywords = build_definition_set(source_path, module_extension)
    if keywords:
        def_keywords = list(def_keywords) + list(keywords)
    return Tokenizer(operators, pairs, comments, def_keywords).tokenize(source, start_pos)


def tokenize_lines(
    lines,
    source_path=None,
    module_extension=".mydef",
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    """複数行をまとめてトークン化する。"""
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


def _advance_pos_by_text(start_pos: PositionTuple, text: str) -> PositionTuple:
    """文字列を読み込んだ後の位置情報を計算する。"""
    idx, line, col = start_pos
    idx += len(text)
    if "\n" in text:
        parts = text.split("\n")
        line += len(parts) - 1
        col = len(parts[-1]) + 1
    else:
        col += len(text)
    return (idx, line, col)


def tokenize_line(
    line,
    operators=None,
    pairs=None,
    comments=None,
    keywords=None,
    start_pos: PositionTuple = (0, 1, 1),
):
    """1行だけトークン化し、次の開始位置も返す。"""
    tokens = tokenize(
        line,
        operators=operators,
        pairs=pairs,
        comments=comments,
        keywords=keywords,
        start_pos=start_pos,
    )
    next_pos = _advance_pos_by_text(start_pos, line)
    return tokens, next_pos


if __name__ == "__main__":
    sample = "var $x = foo + 'bar' // comment\nx <- y"
    for t in tokenize(sample):
        print(t)
