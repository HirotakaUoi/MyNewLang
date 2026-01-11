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

`pair('<open>', '<close>')` はブロックとして扱いません。  
構文で定義されていない `pair` が現れた場合はエラーになります。

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

## 空白の扱い

構文解析では空白をトークンとして扱いません。  
字句解析段階で空白は読み飛ばされる前提です。  
空白判定は `str.isspace()` に従います。

## 対話モードの履歴

- 実行ファイル名と同名の `.history` を実行ディレクトリに保存します
- 最大100件の履歴を保持します
