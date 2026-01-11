#!/usr/bin/python3
# coding=utf-8

from MyNewTestPaser import ParserSession, format_tuple_tree


def main():
    session = ParserSession()
    lines = [
        "var a;\n",
        "a = 1 + 2;\n",
        "if (a < 10) {\n",
        "  a = a + 1;\n",
        "}\n",
        "writeln(a);\n",
    ]
    for line in lines:
        ast_list, error = session.feed_line(line)
        for ast in ast_list:
            print(format_tuple_tree(ast))
        if error:
            print("Parse failed:", error)


if __name__ == "__main__":
    main()
