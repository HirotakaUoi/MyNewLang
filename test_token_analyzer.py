#!/usr/bin/python3
# coding=utf-8

import os
import tempfile
import unittest

from MyNewTokenAnalyzer import (
    TokenizeError,
    Tokenizer,
    build_definition_set,
    load_definitions_from_file,
    parse_definition_lines,
    tokenize,
    tokenize_with_definitions,
)


def token_tuples(tokens):
    return [
        (
            t[0],
            t[1],
            t[2][1],
            t[2][2],
            t[3][1],
            t[3][2],
        )
        for t in tokens
    ]


class TokenizerTests(unittest.TestCase):
    def test_identifiers(self):
        src = "abc _a a1 $x9 'bar'"
        tokens = tokenize(src)
        values = [t[1] for t in tokens]
        self.assertEqual(values, ["abc", "_a", "a1", "$x9", "'bar'"])

    def test_string_literal(self):
        src = '"hello"'
        tokens = tokenize(src)
        self.assertEqual([(tokens[0][0], tokens[0][1])], [("STRING", "hello")])

    def test_unicode_identifier(self):
        src = "猫😀_1"
        tokens = tokenize(src)
        self.assertEqual([(t[0], t[1]) for t in tokens], [("IDENT", "猫😀_1")])

    def test_number_literal(self):
        src = "12 3E5"
        tokens = tokenize(src)
        self.assertEqual([(t[0], t[1]) for t in tokens], [
            ("NUMBER", 12),
            ("NUMBER", 300000),
        ])

    def test_comment_included(self):
        src = "a //c\nb"
        tokens = tokenize(src)
        self.assertEqual([t[0] for t in tokens], ["IDENT", "COMMENT", "IDENT"])
        self.assertEqual(tokens[1][1], "//c")
        self.assertEqual((tokens[1][2][1], tokens[1][2][2]), (1, 3))

    def test_operator_longest_match(self):
        ops = ["<", "<-", "<="]
        t = Tokenizer(ops)
        src = "x <- y < z <= w"
        tokens = t.tokenize(src)
        values = [t[1] for t in tokens if t[0] == "OP"]
        self.assertEqual(values, ["<-", "<", "<="])

    def test_operator_fallback(self):
        ops = ["+", "-"]
        t = Tokenizer(ops)
        tokens = t.tokenize("@")
        self.assertEqual([(tokens[0][0], tokens[0][1])], [("PUNC", "@")])

    def test_positions(self):
        src = "a\n  b"
        tokens = tokenize(src)
        self.assertEqual(token_tuples(tokens), [
            ("IDENT", "a", 1, 1, 1, 2),
            ("IDENT", "b", 2, 3, 2, 4),
        ])

    def test_operator_file(self):
        lines = [
            "op('%%', 40, xfy).\n",
            "op('##', 40, xfy). % comment\n",
            "pair('begin','end').\n",
            "comment('//').\n",
            "keyword('for').\n",
        ]
        ops, pairs, comments, keywords = parse_definition_lines(lines)
        self.assertEqual(ops, ["%%", "##"])
        self.assertEqual(pairs, [("begin", "end")])
        self.assertEqual(comments, ["//"])
        self.assertEqual(keywords, ["for"])

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mod.mydef")
            with open(path, "w", encoding="utf-8") as f:
                f.write("op('%%', 40, xfy).\n")
            ops2, pairs2, comments2, keywords2 = load_definitions_from_file(path)
        self.assertEqual(ops2, ["%%"])
        self.assertEqual(pairs2, [])
        self.assertEqual(comments2, [])
        self.assertEqual(keywords2, [])

    def test_definition_file_loading(self):
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "sample.mylang")
            op_path = os.path.join(td, "sample.mydef")
            with open(op_path, "w", encoding="utf-8") as f:
                f.write("op('%%', 40, xfy).\n")
            with open(src_path, "w", encoding="utf-8") as f:
                f.write("a %% b\n")
            tokens = tokenize_with_definitions("a %% b", src_path)
            ops, pairs, comments, keywords = build_definition_set(src_path)
            self.assertIn("%%", ops)
            self.assertIn("//", comments)
            self.assertIn("for", keywords)
            self.assertEqual([(t[0], t[1]) for t in tokens], [
                ("IDENT", "a"),
                ("OP", "%%"),
                ("IDENT", "b"),
            ])

    def test_keyword_prefix_error(self):
        t = Tokenizer(keywords=["for", "while"])
        with self.assertRaises(TokenizeError):
            t.tokenize("format")

    def test_keyword_token(self):
        tokens = tokenize("for")
        self.assertEqual([(t[0], t[1]) for t in tokens], [("KEYWORD", "for")])


if __name__ == "__main__":
    unittest.main()
