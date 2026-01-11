# MyNewTestPaser 文法仕様（概要）

この文書は `MyNewTestPaser.py` が現在受理できる文法仕様の概要です。

## プログラム

```
program <IDENT> <block>
```

## ブロック

```
<block> ::= "{" <statements> "}"
```

`pair('<open>', '<close>')` が定義されている場合は、その開閉トークンもブロックとして扱います  
（ただし `()` と `[]` はブロック対象外）。

## 文（statement）

```
<statements> ::= [ <expression> ] ( ";" [ <expression> ] )*
```

## 式（expression）

式は演算子の優先順位・結合性を考慮して解析されます。

### 主な構文要素

- `if (...) <expression> [else <expression>]`
- `while (...) <expression>`
- `for ( <expr?> ; <expr?> ; <expr?> ) <expr?>`
- `defun <IDENT>(<params?>) <block>`
- `var <decl> ( , <decl> )*`
- `break`
- `return [ <expression> ]`
- `thread <funcall>`
- `await <array_or_var> <- <thread>`
- `wait` / `wait(<expression>)`

## 変数/配列

```
<array> ::= <IDENT> "[" <index_list> "]"
<index_list> ::= <expr?> ( "," <expr?> )*
```

空の添字は `None` として扱われます（例: `a[]`, `a[, ]`）。

## 関数呼び出し

```
<funcall> ::= <IDENT> "(" <args?> ")"
```

## リテラル

- 数値リテラル（無限長整数）
- 文字列リテラル `"..."`（`"` を含まない）

## 演算子

標準演算子は `MyNewTestPaser.py` 内の `_standard_ops()` に基づきます。  
`.mydef` の `op('<lexeme>', <prec>, <assoc>).` は上書きとして反映されます。

## トークン

字句仕様は `TokenizerSpec.md` を参照してください。
