#!/usr/bin/python3
# coding=utf-8

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

from MyNewTokenAnalyzer_Functional import (
    build_definition_set,
    tokenize_line,
    tokenize_with_definitions,
    tokenize_lines,
)

DEBUG = bool(os.environ.get("MYNEW_DEBUG"))


def _debug(message):
    if DEBUG:
        print(message)


class ParseError(Exception):
    def __init__(self, message, token=None):
        super().__init__(message)
        self.token = token


def _format_token(tok):
    if tok is None:
        return "EOF"
    kind, value, start, end = tok
    return f"{kind} {value!r} at {start}"

def _make_snippet(source, pos_tuple, width=80):
    idx, line, col = pos_tuple
    lines = source.splitlines()
    if line - 1 < 0 or line - 1 >= len(lines):
        return ""
    text = lines[line - 1]
    if len(text) > width:
        start = max(0, col - 1 - width // 2)
        end = min(len(text), start + width)
        view = text[start:end]
        caret_pos = col - 1 - start
    else:
        view = text
        caret_pos = col - 1
    caret_line = " " * max(0, caret_pos) + "^"
    return f"{view}\n{caret_line}"


def _strip_comment(line):
    """mydef の行から % コメントを除去する。"""
    # % コメントを除去する（クォート内の % は除外）
    in_quote = False
    for idx, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "%" and not in_quote:
            return line[:idx].strip()
    return line.strip()


def _parse_op_defs(lines):
    """mydef の op/3 を抽出して演算子定義を返す。"""
    # op('<lexeme>', <precedence>, <assoc>).
    result = []
    for raw in lines:
        line = _strip_comment(raw)
        if not line or not line.endswith("."):
            continue
        body = line[:-1].strip()
        if not (body.startswith("op(") and body.endswith(")")):
            continue
        inner = body[3:-1].strip()
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 3:
            continue
        lexeme, prec_s, assoc = parts
        if lexeme.startswith("'") and lexeme.endswith("'") and len(lexeme) >= 2:
            lexeme = lexeme[1:-1]
        try:
            prec = int(prec_s)
        except ValueError:
            continue
        result.append((lexeme, prec, assoc))
    return result


def _load_module_ops(source_path, module_extension=".mydef"):
    """同名 .mydef から演算子定義を読み込む。"""
    base_dir = os.path.dirname(source_path)
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    module_path = os.path.join(base_dir, base_name + module_extension)
    if not os.path.isfile(module_path):
        return []
    with open(module_path, "r", encoding="utf-8") as f:
        return _parse_op_defs(f.readlines())


def _standard_ops():
    """標準演算子の優先順位・結合性を返す。"""
    # 主要な演算子の優先順位と結合性（高いほど強い）
    return [
        ("=", 10, "xfy"),
        ("+=", 10, "xfy"),
        ("-=", 10, "xfy"),
        ("*=", 10, "xfy"),
        ("/=", 10, "xfy"),
        ("%=", 10, "xfy"),
        ("<-", 10, "xfy"),
        ("||", 20, "yfx"),
        ("&&", 30, "yfx"),
        ("==", 40, "xfx"),
        ("!=", 40, "xfx"),
        ("<", 50, "xfx"),
        (">", 50, "xfx"),
        ("<=", 50, "xfx"),
        (">=", 50, "xfx"),
        ("+", 60, "yfx"),
        ("-", 60, "yfx"),
        ("*", 70, "yfx"),
        ("/", 70, "yfx"),
        ("%", 70, "yfx"),
        ("^", 80, "xfy"),
        ("!", 90, "fy"),
        ("++", 90, "fy"),
        ("--", 90, "fy"),
        ("++", 95, "xf"),
        ("--", 95, "xf"),
    ]


def _build_operator_table(source_path=None, module_extension=".mydef"):
    """演算子の優先順位表（infix/prefix/postfix）を構築する。"""
    op_specs = list(_standard_ops())
    if source_path:
        for lex, prec, assoc in _load_module_ops(source_path, module_extension):
            op_specs.append((lex, prec, assoc))

    infix: Dict[str, Tuple[int, str]] = {}
    prefix: Dict[str, Tuple[int, str]] = {}
    postfix: Dict[str, Tuple[int, str]] = {}

    for lex, prec, assoc in op_specs:
        if assoc in ("xfy", "yfx", "xfx"):
            infix[lex] = (prec, assoc)
        elif assoc in ("fx", "fy"):
            prefix[lex] = (prec, assoc)
        elif assoc in ("xf", "yf"):
            postfix[lex] = (prec, assoc)
    return infix, prefix, postfix


def node(node_type: str, **fields):
    """ASTノード（辞書）を生成する。"""
    data = {"type": node_type}
    data.update(fields)
    return data


_NODE_FIELDS = {
    "PROGRAM": ["name", "body"],
    "INT": ["value"],
    "STRING": ["value"],
    "VAR": ["name"],
    "DEFVAR": ["items"],
    "ARRAY": ["name", "dims", "indices"],
    "ARRAYPARAM": ["name", "dims"],
    "BLOCK": ["items"],
    "IF": ["cond", "then", "else_"],
    "WHILE": ["cond", "body"],
    "FOR": ["init", "cond", "update", "body"],
    "BREAK": ["value"],
    "DEFUN": ["name", "params", "body"],
    "FUNCALL": ["name", "args"],
    "RETURN": ["value"],
    "THREAD": ["name", "args"],
    "AWAIT": ["target", "thread"],
    "WAIT": ["value"],
    "INCF": ["value"],
    "INCB": ["value"],
    "DECF": ["value"],
    "DECB": ["value"],
    "NOT": ["value"],
    "SADD": ["value"],
    "SMINS": ["value"],
    "EQ": ["left", "right"],
    "NEQ": ["left", "right"],
    "LTE": ["left", "right"],
    "GTE": ["left", "right"],
    "AND": ["left", "right"],
    "OR": ["left", "right"],
    "SBTADD": ["left", "right"],
    "SBTMI": ["left", "right"],
    "SBTML": ["left", "right"],
    "SBTDV": ["left", "right"],
    "SBTMD": ["left", "right"],
    "SUBST": ["left", "right"],
    "EXP": ["left", "right"],
    "ADD": ["left", "right"],
    "MINUS": ["left", "right"],
    "MUL": ["left", "right"],
    "DIV": ["left", "right"],
    "MOD": ["left", "right"],
    "GT": ["left", "right"],
    "LT": ["left", "right"],
    "OP": ["op", "left", "right"],
    "UOP": ["op", "value"],
    "POSTOP": ["op", "value"],
}


def to_tuple(value):
    """AST辞書をタプル形式に変換する。"""
    if value is None:
        return None
    if isinstance(value, list):
        return [to_tuple(v) for v in value]
    if isinstance(value, dict):
        node_type = value.get("type")
        fields = _NODE_FIELDS.get(node_type)
        if fields is None:
            raise ValueError(f"Unknown node type: {node_type}")
        return tuple([node_type] + [to_tuple(value.get(f)) for f in fields])
    return value


@dataclass
class TokenStream:
    tokens: List[Tuple[str, Any, Tuple[int, int, int], Tuple[int, int, int]]]
    index: int = 0

    def _skip_comments(self):
        """COMMENT トークンを読み飛ばす。"""
        while self.index < len(self.tokens) and self.tokens[self.index][0] == "COMMENT":
            self.index += 1

    def peek(self, offset=0):
        """現在位置のトークンを参照する（消費しない）。"""
        self._skip_comments()
        idx = self.index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self):
        """現在位置のトークンを消費する。"""
        self._skip_comments()
        tok = self.peek()
        if tok is None:
            return None
        self.index += 1
        return tok

    def match(self, kind=None, value=None):
        """条件に合うならトークンを消費して返す。"""
        tok = self.peek()
        if tok is None:
            return None
        if kind is not None and tok[0] != kind:
            return None
        if value is not None and tok[1] != value:
            return None
        self.index += 1
        return tok

    def expect(self, kind=None, value=None):
        """条件に合うトークンが無ければエラー。"""
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected EOF", None)
        if kind is not None and tok[0] != kind:
            raise ParseError(f"Expected {kind}", tok)
        if value is not None and tok[1] != value:
            raise ParseError(f"Expected {value}", tok)
        self.index += 1
        return tok

    def match_punc(self, value):
        """PUNC の一致を試行する。"""
        return self.match("PUNC", value)

    def expect_punc(self, value):
        """PUNC を強制的に期待する。"""
        return self.expect("PUNC", value)

    def match_lexeme(self, value):
        """OP/IDENT/KEYWORD をまとめて一致判定する。"""
        tok = self.peek()
        if tok is None:
            return None
        if tok[0] == "OP" and tok[1] == value:
            self.index += 1
            return tok
        if tok[0] in ("KEYWORD", "IDENT") and tok[1] == value:
            self.index += 1
            return tok
        return None

    def expect_lexeme(self, value):
        """OP/IDENT/KEYWORD をまとめて期待する。"""
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected EOF", None)
        if tok[0] == "OP" and tok[1] == value:
            self.index += 1
            return tok
        if tok[0] in ("KEYWORD", "IDENT") and tok[1] == value:
            self.index += 1
            return tok
        raise ParseError(f"Expected {value}", tok)

    def expect_word(self, word):
        """キーワードまたは識別子の一致を期待する。"""
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected EOF", None)
        if tok[0] in ("KEYWORD", "IDENT") and tok[1] == word:
            self.index += 1
            return tok
        raise ParseError(f"Expected keyword {word}", tok)

    def is_word(self, word):
        """キーワード/識別子の一致を判定する。"""
        tok = self.peek()
        return tok is not None and tok[0] in ("KEYWORD", "IDENT") and tok[1] == word


_BIN_OP_MAP = {
    "==": "EQ",
    "!=": "NEQ",
    "<=": "LTE",
    ">=": "GTE",
    "&&": "AND",
    "||": "OR",
    "+=": "SBTADD",
    "-=": "SBTMI",
    "*=": "SBTML",
    "/=": "SBTDV",
    "%=": "SBTMD",
    "=": "SUBST",
    "^": "EXP",
    "+": "ADD",
    "-": "MINUS",
    "*": "MUL",
    "/": "DIV",
    "%": "MOD",
    ">": "GT",
    "<": "LT",
}

_UNARY_OP_MAP = {
    "+": "SADD",
    "-": "SMINS",
    "!": "NOT",
}


class Parser:
    def __init__(self, tokens, source_path=None, module_extension=".mydef"):
        """トークン列からASTを構築するパーサ。"""
        self.stream = TokenStream(tokens)
        self.infix_ops, self.prefix_ops, self.postfix_ops = _build_operator_table(
            source_path, module_extension
        )
        self.pairs = []

    def _match_block_open(self):
        """ブロック開始を確認し、対応する閉じを返す。"""
        if self.stream.match("OP", "{"):
            return "{", "}"
        return None

    def parse(self):
        """program を先頭から解析する。"""
        prog = self.parse_program()
        if self.stream.peek() is not None:
            raise ParseError("Unexpected token after program", self.stream.peek())
        return prog

    def parse_program(self):
        """program 構文を解析する。"""
        self.stream.expect_word("program")
        name_tok = self.stream.expect("IDENT")
        body = self.parse_block()
        return node("PROGRAM", name=name_tok[1], body=body)

    def parse_block(self):
        """ブロック（{ ... }）を解析する。"""
        pair = self._match_block_open()
        if pair is None:
            raise ParseError("Expected block start", self.stream.peek())
        _open_lexeme, close_lexeme = pair
        items = self.parse_statements(close_lexeme)
        _debug(f"parse_block: expect close={close_lexeme}, next={self.stream.peek()}")
        self.stream.expect_lexeme(close_lexeme)
        return node("BLOCK", items=items)

    def parse_statements(self, close_lexeme=None):
        """セミコロン区切りの文リストを解析する。"""
        items = []
        if self.stream.peek() is None:
            return items
        if close_lexeme and self.stream.peek() and self.stream.peek()[1] == close_lexeme:
            return items
        expr = self.parse_expression_optional(close_lexeme)
        if expr is not None:
            items.append(expr)
        while self.stream.match_punc(";"):
            expr = self.parse_expression_optional(close_lexeme)
            if expr is not None:
                items.append(expr)
        return items

    def parse_expression_optional(self, close_lexeme=None):
        """空式を許す式解析。"""
        tok = self.stream.peek()
        if tok is None:
            return None
        if tok[0] == "PUNC" and tok[1] == ";":
            return None
        if close_lexeme and tok[1] == close_lexeme:
            return None
        return self.parse_expression()

    def parse_expression(self, min_prec=0):
        """優先順位付きの式を解析する。"""
        # Pratt風の優先順位パース（prefix/infix/postfix）
        left = self.parse_prefix()
        while True:
            tok = self.stream.peek()
            if tok is None or tok[0] != "OP":
                break
            op = tok[1]

            if op in self.postfix_ops:
                prec, assoc = self.postfix_ops[op]
                if prec < min_prec:
                    break
                self.stream.consume()
                if op == "++":
                    left = node("INCB", value=left)
                elif op == "--":
                    left = node("DECB", value=left)
                else:
                    left = node("POSTOP", op=op, value=left)
                continue

            if op not in self.infix_ops:
                break
            prec, assoc = self.infix_ops[op]
            if prec < min_prec:
                break
            self.stream.consume()
            if assoc == "yfx":
                next_min = prec + 1
            elif assoc == "xfx":
                next_min = prec + 1
            else:
                next_min = prec
            right = self.parse_expression(next_min)
            left = self.make_binary(op, left, right)
        return left

    def parse_prefix(self):
        """前置演算子またはコア式を解析する。"""
        tok = self.stream.peek()
        if tok is not None and tok[0] == "OP" and tok[1] in self.prefix_ops:
            op = tok[1]
            prec, _assoc = self.prefix_ops[op]
            self.stream.consume()
            right = self.parse_expression(prec)
            if op == "++":
                return node("INCF", value=right)
            if op == "--":
                return node("DECF", value=right)
            op_type = _UNARY_OP_MAP.get(op)
            if op_type:
                return node(op_type, value=right)
            return node("UOP", op=op, value=right)
        return self.parse_core_expression()

    def parse_core_expression(self):
        """構文キーワードやブロックなどの中核構文を解析する。"""
        if self.stream.is_word("await"):
            return self.parse_await()
        if self.stream.is_word("thread"):
            return self.parse_thread()
        if self.stream.is_word("wait"):
            return self.parse_wait()
        if self.stream.is_word("if"):
            return self.parse_if()
        if self.stream.is_word("while"):
            return self.parse_while()
        if self.stream.is_word("for"):
            return self.parse_for()
        if self.stream.is_word("defun"):
            return self.parse_defun()
        if self.stream.is_word("var"):
            return self.parse_defvar()
        if self.stream.is_word("break"):
            self.stream.consume()
            return node("BREAK", value=None)
        if self.stream.is_word("return"):
            self.stream.consume()
            value = self.parse_expression_optional()
            return node("RETURN", value=value)
        if self.stream.peek() and self.stream.peek()[0] == "OP" and self.stream.peek()[1] == "{":
            return self.parse_block()
        return self.parse_factor()

    def parse_if(self):
        """if 文を解析する。"""
        self.stream.expect_word("if")
        self.stream.expect("OP", "(")
        cond = self.parse_expression()
        self.stream.expect("OP", ")")
        then = self.parse_expression()
        else_expr = None
        if self.stream.is_word("else"):
            self.stream.consume()
            else_expr = self.parse_expression()
        return node("IF", cond=cond, then=then, else_=else_expr)

    def parse_while(self):
        """while 文を解析する。"""
        self.stream.expect_word("while")
        self.stream.expect("OP", "(")
        cond = self.parse_expression_optional()
        self.stream.expect("OP", ")")
        body = self.parse_expression_optional()
        return node("WHILE", cond=cond, body=body)

    def parse_for(self):
        """for 文を解析する。"""
        self.stream.expect_word("for")
        self.stream.expect("OP", "(")
        init = self.parse_expression_optional()
        self.stream.expect_punc(";")
        cond = self.parse_expression_optional()
        self.stream.expect_punc(";")
        update = self.parse_expression_optional()
        self.stream.expect("OP", ")")
        body = self.parse_expression_optional()
        return node("FOR", init=init, cond=cond, update=update, body=body)

    def parse_defun(self):
        """関数定義を解析する。"""
        self.stream.expect_word("defun")
        name_tok = self.stream.expect("IDENT")
        self.stream.expect("OP", "(")
        params = []
        if not self.stream.match("OP", ")"):
            params.append(self.parse_func_param())
            while self.stream.match_punc(","):
                params.append(self.parse_func_param())
            self.stream.expect("OP", ")")
        body = self.parse_block()
        return node("DEFUN", name=name_tok[1], params=params, body=body)

    def parse_func_param(self):
        """関数仮引数を解析する。"""
        name_tok = self.stream.expect("IDENT")
        if self.stream.match("OP", "["):
            count = 0
            while self.stream.match_punc(","):
                count += 1
            self.stream.expect("OP", "]")
            return node("ARRAYPARAM", name=name_tok[1], dims=count + 1)
        return node("VAR", name=name_tok[1])

    def parse_defvar(self):
        """var 宣言を解析する。"""
        self.stream.expect_word("var")
        items = [self.parse_array_or_var()]
        while self.stream.match_punc(","):
            items.append(self.parse_array_or_var())
        return node("DEFVAR", items=items)

    def parse_thread(self):
        """thread 呼び出しを解析する。"""
        self.stream.expect_word("thread")
        call = self.parse_funcall()
        return node("THREAD", name=call["name"], args=call["args"])

    def parse_wait(self):
        """wait 構文を解析する。"""
        self.stream.expect_word("wait")
        if self.stream.match("OP", "("):
            if self.stream.match("OP", ")"):
                return node("WAIT", value=[])
            value = self.parse_expression()
            self.stream.expect("OP", ")")
            return node("WAIT", value=value)
        value = self.parse_expression_optional()
        return node("WAIT", value=value if value is not None else [])

    def parse_await(self):
        """await 構文を解析する。"""
        self.stream.expect_word("await")
        target = self.parse_array_or_var()
        self.stream.expect("OP", "<-")
        thread_expr = self.parse_thread()
        return node("AWAIT", target=target, thread=thread_expr)

    def parse_factor(self):
        """リテラル・変数・呼び出し・括弧式などを解析する。"""
        tok = self.stream.peek()
        if tok is None:
            raise ParseError("Unexpected EOF", None)
        if tok[0] == "NUMBER":
            self.stream.consume()
            return node("INT", value=tok[1])
        if tok[0] == "STRING":
            self.stream.consume()
            return node("STRING", value=tok[1])
        if tok[0] in ("IDENT", "KEYWORD"):
            name_tok = self.stream.consume()
            if self.stream.match("OP", "("):
                self.stream.index -= 1
                return self.parse_funcall_from(name_tok[1])
            if self.stream.match("OP", "["):
                self.stream.index -= 1
                return self.parse_array_from(name_tok[1])
            return node("VAR", name=name_tok[1])
        if tok[0] == "OP" and tok[1] == "(":
            self.stream.consume()
            expr = self.parse_expression()
            self.stream.expect("OP", ")")
            return expr
        _debug(f"parse_factor: unexpected token {tok}")
        raise ParseError("Unexpected token in factor", tok)

    def parse_funcall(self):
        """関数呼び出しを解析する。"""
        name_tok = self.stream.expect("IDENT")
        return self.parse_funcall_from(name_tok[1])

    def parse_funcall_from(self, name):
        """関数名を受け取って呼び出しを解析する。"""
        self.stream.expect("OP", "(")
        args = []
        if not self.stream.match("OP", ")"):
            _debug(f"parse_funcall_from({name}): first arg token {self.stream.peek()}")
            args.append(self.parse_expression())
            while self.stream.match_punc(","):
                _debug(f"parse_funcall_from({name}): next arg token {self.stream.peek()}")
                args.append(self.parse_expression())
            self.stream.expect("OP", ")")
        return node("FUNCALL", name=name, args=args)

    def parse_array_or_var(self):
        """配列アクセスか変数参照を解析する。"""
        name_tok = self.stream.expect("IDENT")
        if self.stream.peek() and self.stream.peek()[0] == "OP" and self.stream.peek()[1] == "[":
            return self.parse_array_from(name_tok[1])
        return node("VAR", name=name_tok[1])

    def parse_array_from(self, name):
        """配列アクセスを解析する。"""
        self.stream.expect("OP", "[")
        # 空添字を None として保持する（a[] / a[, ] など）
        indices = []
        expecting_expr = True
        while True:
            tok = self.stream.peek()
            if tok is None:
                raise ParseError("Unexpected EOF in array index", tok)
            if tok[0] == "OP" and tok[1] == "]":
                if expecting_expr:
                    indices.append(None)
                self.stream.consume()
                break
            if tok[0] == "PUNC" and tok[1] == ",":
                if expecting_expr:
                    indices.append(None)
                self.stream.consume()
                expecting_expr = True
                continue
            indices.append(self.parse_expression())
            expecting_expr = False
        return node("ARRAY", name=name, dims=len(indices), indices=indices)

    def make_binary(self, op, left, right):
        """2項演算のASTノードを生成する。"""
        op_type = _BIN_OP_MAP.get(op)
        if op_type:
            return node(op_type, left=left, right=right)
        return node("OP", op=op, left=left, right=right)


def parse_tokens(tokens, source_path=None, module_extension=".mydef"):
    """トークン列からASTタプルを生成する。"""
    parser = Parser(tokens, source_path=source_path, module_extension=module_extension)
    return to_tuple(parser.parse())


def parse_tokens_ast(tokens, source_path=None, module_extension=".mydef"):
    """トークン列からAST辞書を生成する。"""
    parser = Parser(tokens, source_path=source_path, module_extension=module_extension)
    return parser.parse()


def parse(source, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
    """文字列入力をバッチ解析する。"""
    tokens = tokenize_with_definitions(
        source, source_path=source_path or "", module_extension=module_extension, start_pos=start_pos
    )
    return parse_tokens(tokens, source_path=source_path, module_extension=module_extension)


def parse_lines(lines, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
    """行リスト入力をバッチ解析する。"""
    tokens = tokenize_lines(
        lines, source_path=source_path, module_extension=module_extension, start_pos=start_pos
    )
    return parse_tokens(tokens, source_path=source_path, module_extension=module_extension)


def parse_file(file_path, module_extension=".mydef", start_pos=(0, 1, 1)):
    """ファイル入力をバッチ解析する。"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return parse_lines(lines, source_path=file_path, module_extension=module_extension, start_pos=start_pos)


def parse_ast(source, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
    """文字列入力からAST辞書を生成する。"""
    tokens = tokenize_with_definitions(
        source, source_path=source_path or "", module_extension=module_extension, start_pos=start_pos
    )
    return parse_tokens_ast(tokens, source_path=source_path, module_extension=module_extension)


def _tree_lines(value, indent=0):
    """辞書ASTを木構造の行列に変換する。"""
    pad = "  " * indent
    if value is None:
        return [pad + "None"]
    if isinstance(value, dict):
        node_type = value.get("type", "UNKNOWN")
        lines = [pad + str(node_type)]
        fields = _NODE_FIELDS.get(node_type, [])
        for field in fields:
            child = value.get(field)
            if isinstance(child, list):
                lines.append(pad + "  " + field + ":")
                if not child:
                    lines.append(pad + "    " + "[]")
                for item in child:
                    lines.extend(_tree_lines(item, indent + 3))
            else:
                lines.append(pad + "  " + field + ":")
                lines.extend(_tree_lines(child, indent + 3))
        return lines
    if isinstance(value, list):
        if not value:
            return [pad + "[]"]
        lines = [pad + "[]"]
        for item in value:
            lines.extend(_tree_lines(item, indent + 1))
        return lines
    return [pad + repr(value)]


def format_tree(ast):
    """辞書ASTを木構造で表示する。"""
    return "\n".join(_tree_lines(ast))


def _tuple_tree_lines(value, indent=0):
    """タプルASTを木構造の行列に変換する。"""
    pad = "  " * indent
    if value is None:
        return [pad + "None"]
    if isinstance(value, list):
        if not value:
            return [pad + "[]"]
        lines = []
        for item in value:
            lines.extend(_tuple_tree_lines(item, indent + 1))
        return lines
    if isinstance(value, tuple) and value:
        node_type = value[0]
        fields = _NODE_FIELDS.get(node_type)
        if fields is None:
            return [pad + repr(value)]
        inline_parts = []
        child_fields = []
        for field, child in zip(fields, value[1:]):
            is_child = False
            if isinstance(child, list):
                is_child = True
            elif isinstance(child, tuple) and child:
                child_type = child[0]
                if child_type in _NODE_FIELDS:
                    is_child = True
            if is_child:
                child_fields.append((field, child))
            else:
                inline_parts.append(f"{field}={repr(child)}")
        header = str(node_type)
        if inline_parts:
            header += " " + " ".join(inline_parts)
        lines = [pad + header]
        for field, child in child_fields:
            if isinstance(child, tuple) and child and child[0] in _NODE_FIELDS:
                nested_lines = _tuple_tree_lines(child, indent + 2)
                if nested_lines:
                    first = nested_lines[0].lstrip()
                    lines.append(pad + "  " + field + ": " + first)
                    base = "  " * (indent + 2)
                    for extra in nested_lines[1:]:
                        tail = extra
                        if tail.startswith(base):
                            tail = tail[len(base):]
                        lines.append(pad + "    " + tail)
                continue
            if isinstance(child, list) and not child:
                lines.append(pad + "  " + field + ": []")
                continue
            lines.append(pad + "  " + field + ":")
            lines.extend(_tuple_tree_lines(child, indent + 2))
        return lines
    return [pad + repr(value)]


def format_tuple_tree(ast_tuple):
    """タプルASTを木構造で表示する。"""
    return "\n".join(_tuple_tree_lines(ast_tuple))


class ParserSession:
    def __init__(self, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
        """行単位入力の解析セッションを初期化する。"""
        self.source_path = source_path
        self.module_extension = module_extension
        self.tokens = []
        self.source_text = ""
        self.next_pos = start_pos
        self.operators, self.pairs, self.comments, self.keywords = build_definition_set(
            source_path, module_extension
        )

    def feed_line(self, line, parse_mode="stmt"):
        # 行単位で字句解析し、確定した文のASTのみを返す
        tokens, self.next_pos = tokenize_line(
            line,
            operators=self.operators,
            pairs=self.pairs,
            comments=self.comments,
            keywords=self.keywords,
            start_pos=self.next_pos,
        )
        self.tokens.extend(tokens)
        self.source_text += line
        results = []
        error_info = None

        while self.tokens:
            parser = Parser(self.tokens, source_path=self.source_path, module_extension=self.module_extension)
            try:
                if parse_mode == "program":
                    prog = parser.parse_program()
                    consumed = parser.stream.index
                    results.append(to_tuple(prog))
                    self.tokens = self.tokens[consumed:]
                    continue

                expr = parser.parse_expression_optional()
                if expr is None:
                    break
                if parse_mode == "stmt" and parser.stream.match_punc(";"):
                    consumed = parser.stream.index
                    results.append(to_tuple(expr))
                    self.tokens = self.tokens[consumed:]
                    continue
                if parse_mode == "expr" or parser.stream.peek() is None:
                    consumed = parser.stream.index
                    results.append(to_tuple(expr))
                    self.tokens = self.tokens[consumed:]
                    continue
                error_info = f"Unexpected token: {_format_token(parser.stream.peek())}"
                self.tokens = []
                break
            except ParseError as exc:
                if exc.token is None:
                    break
                error_info = f"{exc} at {_format_token(exc.token)}"
                snippet = _make_snippet(self.source_text, exc.token[2])
                if snippet:
                    error_info += "\n" + snippet
                self.tokens = []
                break
        return results, error_info


def batch_parse_text(source_text, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
    """文字列入力のバッチ解析結果を表示する。"""
    # 一括入力のパース結果を表示する
    try:
        ast_tuple = parse(source_text, source_path=source_path, module_extension=module_extension, start_pos=start_pos)
        print(format_tuple_tree(ast_tuple))
    except Exception as exc:
        print(f"Parse failed: {exc}")
        try:
            tokens = tokenize_with_definitions(source_text, source_path=source_path or "")
        except Exception as lex_exc:
            print(f"Lexing failed: {lex_exc}")
            return
        err_tok = None
        if isinstance(exc, ParseError):
            err_tok = exc.token
        if err_tok is None and tokens:
            err_tok = tokens[-1]
        print(f"Error token: {_format_token(err_tok)}")
        if err_tok is not None:
            snippet = _make_snippet(source_text, err_tok[2])
            if snippet:
                print(snippet)


def batch_parse_file(file_path, module_extension=".mydef", start_pos=(0, 1, 1)):
    """ファイル入力のバッチ解析結果を表示する。"""
    # ファイル入力のバッチ解析
    with open(file_path, "r", encoding="utf-8") as f:
        source_text = f.read()
    batch_parse_text(
        source_text,
        source_path=file_path,
        module_extension=module_extension,
        start_pos=start_pos,
    )


def interactive_parse():
    """対話モードで行単位解析を行う。"""
    # 対話モード: 1行ずつ解析し、確定結果のみ出力する
    session = ParserSession()
    last_line = None
    buffered_source = ""
    buffered_tokens = []
    first_prompt = True
    mode = "interactive"
    parse_mode = "expr"
    while True:
        try:
            if first_prompt:
                print("Input line (empty line = repeat, Ctrl-D = end):", flush=True)
                first_prompt = False
            print(f"({session.next_pos[1]}):> ", end="", flush=True)
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if line == "":
            buffered_source = session.source_text
            buffered_tokens = list(session.tokens)
            if buffered_source or buffered_tokens:
                print(f"Buffered {len(buffered_source)} chars, {len(buffered_tokens)} tokens.")
            break
        if line == "\n":
            if last_line is None:
                continue
            line = last_line
        else:
            last_line = line

        stripped = line.strip()
        if stripped == ":batch":
            mode = "batch"
            print("Mode: batch")
            continue
        if stripped == ":interactive":
            mode = "interactive"
            print("Mode: interactive")
            continue
        if stripped.startswith(":mode"):
            parts = stripped.split()
            if len(parts) == 2 and parts[1] in ("program", "stmt", "expr"):
                parse_mode = parts[1]
                print(f"Parse mode: {parse_mode}")
            else:
                print("Usage: :mode program|stmt|expr")
            continue

        if mode == "interactive":
            results, error_info = session.feed_line(line, parse_mode=parse_mode)
            for ast in results:
                print(format_tuple_tree(ast))
            if error_info:
                print(f"Parse failed: {error_info}")
        else:
            buffered_source += line
            session.source_text = buffered_source
            session.tokens = []
            session.next_pos = (0, 1, 1)
            batch_parse_text(buffered_source)


if __name__ == "__main__":
    import sys
    import os
    import readline

    print("My New Test Parser (Parser Combinator Style)")
    if sys.stdin.isatty():
        history_path = os.path.basename(sys.argv[0]) + ".history"
        try:
            if os.path.isfile(history_path):
                readline.read_history_file(history_path)
        except OSError:
            pass
        readline.set_history_length(100)
        interactive_parse()
        try:
            readline.write_history_file(history_path)
        except OSError:
            pass
    else:
        source_text = sys.stdin.read()
        if source_text:
            batch_parse_text(source_text)
