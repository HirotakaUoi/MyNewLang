#!/usr/bin/python3
# coding=utf-8

from dataclasses import dataclass
import argparse
from typing import Iterable, List, Optional, Tuple

from MyNewTokenAnalyzer import tokenize as tokenize_imp
from MyNewTokenAnalyzer import tokenize_with_definitions as tokenize_imp_defs
from MyNewTokenAnalyzer_Functional import tokenize as tokenize_func
from MyNewTokenAnalyzer_Functional import tokenize_with_definitions as tokenize_func_defs


@dataclass(frozen=True)
class TokenInfo:
    kind: str
    value: object
    start: Tuple[int, int, int]
    end: Tuple[int, int, int]


def _token_info(tokens) -> List[TokenInfo]:
    return [
        TokenInfo(
            t[0],
            t[1],
            t[2],
            t[3],
        )
        for t in tokens
    ]


def compare_sources(sources: Iterable[str]) -> List[str]:
    diffs = []
    for idx, src in enumerate(sources):
        t1 = _token_info(tokenize_imp(src))
        t2 = _token_info(tokenize_func(src))
        if t1 != t2:
            diffs.append(f"#{idx}: mismatch")
    return diffs


def _print_diff(index: int, src: str, source_path: Optional[str] = None):
    if source_path:
        t1 = _token_info(tokenize_imp_defs(src, source_path))
        t2 = _token_info(tokenize_func_defs(src, source_path))
    else:
        t1 = _token_info(tokenize_imp(src))
        t2 = _token_info(tokenize_func(src))
    if t1 == t2:
        return
    print(f"Source #{index} mismatch:")
    print(src)
    print("imperative:", t1)
    print("functional:", t2)


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizer outputs.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source file path; loads matching .mydef definitions if present.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Inline source text.",
    )
    parser.add_argument(
        "--defs-source",
        help="Use this file path to resolve .mydef definitions for --text inputs.",
    )
    args = parser.parse_args()

    if not args.source and not args.text:
        samples = [
            "var $x = foo + 'bar' // comment\nx <- y",
            "a\n  b",
            "12 3E5",
            "x <- y < z <= w",
            "\"hello\" 'id' $x9 abc",
            "a //c\nb",
            "@",
        ]
        diffs = compare_sources(samples)
        if not diffs:
            print("All samples match.")
            return
        for i, src in enumerate(samples):
            _print_diff(i, src)
        return

    sources = []
    for path in args.source:
        with open(path, "r", encoding="utf-8") as f:
            sources.append((path, f.read()))

    for text in args.text:
        sources.append((args.defs_source, text))

    mismatched = False
    for i, (source_path, src) in enumerate(sources):
        if source_path:
            t1 = _token_info(tokenize_imp_defs(src, source_path))
            t2 = _token_info(tokenize_func_defs(src, source_path))
        else:
            t1 = _token_info(tokenize_imp(src))
            t2 = _token_info(tokenize_func(src))
        if t1 != t2:
            mismatched = True
            _print_diff(i, src, source_path)

    if not mismatched:
        print("All sources match.")


if __name__ == "__main__":
    main()
