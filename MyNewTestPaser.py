#!/usr/bin/python3
# coding=utf-8

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Tuple

from MyNewTokenAnalyzer import build_definition_set, tokenize_with_definitions, tokenize_lines


class ParseError(Exception):
    def __init__(self, message, token=None):
        super().__init__(message)
        self.token = token


def _strip_comment(line):
    # % コメントを除去する（クォート内の % は除外）
    in_quote = False
    for idx, ch in enumerate(line):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "%" and not in_quote:
            return line[:idx].strip()
    return line.strip()


def _parse_op_defs(lines):
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
    base_dir = os.path.dirname(source_path)
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    module_path = os.path.join(base_dir, base_name + module_extension)
    if not os.path.isfile(module_path):
        return []
    with open(module_path, "r", encoding="utf-8") as f:
        return _parse_op_defs(f.readlines())


def _standard_ops():
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
    op_specs = {lex: (prec, assoc) for lex, prec, assoc in _standard_ops()}
    if source_path:
        for lex, prec, assoc in _load_module_ops(source_path, module_extension):
            op_specs[lex] = (prec, assoc)

    infix: Dict[str, Tuple[int, str]] = {}
    prefix: Dict[str, Tuple[int, str]] = {}
    postfix: Dict[str, Tuple[int, str]] = {}

    for lex, (prec, assoc) in op_specs.items():
        if assoc in ("xfy", "yfx", "xfx"):
            infix[lex] = (prec, assoc)
        elif assoc in ("fx", "fy"):
            prefix[lex] = (prec, assoc)
        elif assoc in ("xf", "yf"):
            postfix[lex] = (prec, assoc)
    return infix, prefix, postfix


def node(node_type: str, **fields):
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
    "BLOCK": ["items", "begin", "end"],
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
        while self.index < len(self.tokens) and self.tokens[self.index][0] == "COMMENT":
            self.index += 1

    def peek(self, offset=0):
        self._skip_comments()
        idx = self.index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self):
        self._skip_comments()
        tok = self.peek()
        if tok is None:
            return None
        self.index += 1
        return tok

    def match(self, kind=None, value=None):
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
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected EOF", None)
        if kind is not None and tok[0] != kind:
            raise ParseError(f"Expected {kind}", tok)
        if value is not None and tok[1] != value:
            raise ParseError(f"Expected {value}", tok)
        self.index += 1
        return tok

    def match_lexeme(self, value):
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
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected EOF", None)
        if tok[0] in ("KEYWORD", "IDENT") and tok[1] == word:
            self.index += 1
            return tok
        raise ParseError(f"Expected keyword {word}", tok)

    def is_word(self, word):
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
        self.stream = TokenStream(tokens)
        self.infix_ops, self.prefix_ops, self.postfix_ops = _build_operator_table(
            source_path, module_extension
        )
        _ops, pairs, _comments, _keywords = build_definition_set(
            source_path, module_extension
        )
        self.pairs = list(pairs)
        self.block_pairs = self._build_block_pairs(self.pairs)

    def _build_block_pairs(self, pairs):
        # ブロック候補として { } と、定義済みの括弧対から () [] を除いたものを使う
        block_pairs = [("{", "}")]
        for left, right in pairs:
            if (left, right) in (("{", "}"), ("(", ")"), ("[", "]")):
                continue
            block_pairs.append((left, right))
        return block_pairs

    def _begin_tag(self, lexeme):
        return f"BEGIN({lexeme})"

    def _end_tag(self, lexeme):
        return f"END({lexeme})"

    def _match_block_open(self):
        for left, right in self.block_pairs:
            if self.stream.match_lexeme(left):
                return left, right
        return None

    def parse(self):
        prog = self.parse_program()
        if self.stream.peek() is not None:
            raise ParseError("Unexpected token after program", self.stream.peek())
        return prog

    def parse_program(self):
        self.stream.expect_word("program")
        name_tok = self.stream.expect("IDENT")
        body = self.parse_block()
        return node("PROGRAM", name=name_tok[1], body=body)

    def parse_block(self):
        pair = self._match_block_open()
        if pair is None:
            raise ParseError("Expected block start", self.stream.peek())
        open_lexeme, close_lexeme = pair
        items = self.parse_statements(close_lexeme)
        self.stream.expect_lexeme(close_lexeme)
        return node(
            "BLOCK",
            items=items,
            begin=self._begin_tag(open_lexeme),
            end=self._end_tag(close_lexeme),
        )

    def parse_statements(self, close_lexeme=None):
        items = []
        if self.stream.peek() is None:
            return items
        if close_lexeme and self.stream.peek() and self.stream.peek()[1] == close_lexeme:
            return items
        expr = self.parse_expression_optional(close_lexeme)
        if expr is not None:
            items.append(expr)
        while self.stream.match("OP", ";"):
            expr = self.parse_expression_optional(close_lexeme)
            if expr is not None:
                items.append(expr)
        return items

    def parse_expression_optional(self, close_lexeme=None):
        tok = self.stream.peek()
        if tok is None:
            return None
        if tok[0] == "OP" and tok[1] == ";":
            return None
        if close_lexeme and tok[1] == close_lexeme:
            return None
        return self.parse_expression()

    def parse_expression(self, min_prec=0):
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
        if self.stream.peek() and self.stream.peek()[1] in [p[0] for p in self.block_pairs]:
            # ブロック
            return self.parse_block()
        return self.parse_factor()

    def parse_if(self):
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
        self.stream.expect_word("while")
        self.stream.expect("OP", "(")
        cond = self.parse_expression_optional()
        self.stream.expect("OP", ")")
        body = self.parse_expression_optional()
        return node("WHILE", cond=cond, body=body)

    def parse_for(self):
        self.stream.expect_word("for")
        self.stream.expect("OP", "(")
        init = self.parse_expression_optional()
        self.stream.expect("OP", ";")
        cond = self.parse_expression_optional()
        self.stream.expect("OP", ";")
        update = self.parse_expression_optional()
        self.stream.expect("OP", ")")
        body = self.parse_expression_optional()
        return node("FOR", init=init, cond=cond, update=update, body=body)

    def parse_defun(self):
        self.stream.expect_word("defun")
        name_tok = self.stream.expect("IDENT")
        self.stream.expect("OP", "(")
        params = []
        if not self.stream.match("OP", ")"):
            params.append(self.parse_func_param())
            while self.stream.match("OP", ","):
                params.append(self.parse_func_param())
            self.stream.expect("OP", ")")
        body = self.parse_block()
        return node("DEFUN", name=name_tok[1], params=params, body=body)

    def parse_func_param(self):
        name_tok = self.stream.expect("IDENT")
        if self.stream.match("OP", "["):
            count = 0
            while self.stream.match("OP", ","):
                count += 1
            self.stream.expect("OP", "]")
            return node("ARRAYPARAM", name=name_tok[1], dims=count + 1)
        return node("VAR", name=name_tok[1])

    def parse_defvar(self):
        self.stream.expect_word("var")
        items = [self.parse_array_or_var()]
        while self.stream.match("OP", ","):
            items.append(self.parse_array_or_var())
        return node("DEFVAR", items=items)

    def parse_thread(self):
        self.stream.expect_word("thread")
        call = self.parse_funcall()
        return node("THREAD", name=call["name"], args=call["args"])

    def parse_wait(self):
        self.stream.expect_word("wait")
        value = self.parse_expression_optional()
        return node("WAIT", value=value if value is not None else [])

    def parse_await(self):
        self.stream.expect_word("await")
        target = self.parse_array_or_var()
        self.stream.expect("OP", "<-")
        thread_expr = self.parse_thread()
        return node("AWAIT", target=target, thread=thread_expr)

    def parse_factor(self):
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
        raise ParseError("Unexpected token in factor", tok)

    def parse_funcall(self):
        name_tok = self.stream.expect("IDENT")
        return self.parse_funcall_from(name_tok[1])

    def parse_funcall_from(self, name):
        self.stream.expect("OP", "(")
        args = []
        if not self.stream.match("OP", ")"):
            args.append(self.parse_expression())
            while self.stream.match("OP", ","):
                args.append(self.parse_expression())
            self.stream.expect("OP", ")")
        return node("FUNCALL", name=name, args=args)

    def parse_array_or_var(self):
        name_tok = self.stream.expect("IDENT")
        if self.stream.peek() and self.stream.peek()[0] == "OP" and self.stream.peek()[1] == "[":
            return self.parse_array_from(name_tok[1])
        return node("VAR", name=name_tok[1])

    def parse_array_from(self, name):
        self.stream.expect("OP", "[")
        indices = [self.parse_expression()]
        while self.stream.match("OP", ","):
            indices.append(self.parse_expression())
        self.stream.expect("OP", "]")
        return node("ARRAY", name=name, dims=len(indices), indices=indices)

    def make_binary(self, op, left, right):
        op_type = _BIN_OP_MAP.get(op)
        if op_type:
            return node(op_type, left=left, right=right)
        return node("OP", op=op, left=left, right=right)


def parse_tokens(tokens, source_path=None, module_extension=".mydef"):
    parser = Parser(tokens, source_path=source_path, module_extension=module_extension)
    return to_tuple(parser.parse())


def parse(source, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
    tokens = tokenize_with_definitions(
        source, source_path=source_path or "", module_extension=module_extension, start_pos=start_pos
    )
    return parse_tokens(tokens, source_path=source_path, module_extension=module_extension)


def parse_lines(lines, source_path=None, module_extension=".mydef", start_pos=(0, 1, 1)):
    tokens = tokenize_lines(
        lines, source_path=source_path, module_extension=module_extension, start_pos=start_pos
    )
    return parse_tokens(tokens, source_path=source_path, module_extension=module_extension)


def parse_file(file_path, module_extension=".mydef", start_pos=(0, 1, 1)):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return parse_lines(lines, source_path=file_path, module_extension=module_extension, start_pos=start_pos)


if __name__ == "__main__":
    import sys

    print("My New Test Parser (Parser Combinator Style)")
    source_text = sys.stdin.read()
    if source_text:
        ast = parse(source_text)
        print(ast)
