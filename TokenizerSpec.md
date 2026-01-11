# トークナイザ仕様

このドキュメントは `MyNewLang/MyNewTokenAnalyzer.py` の字句解析仕様をまとめたものです。

## トークン種別

- `IDENT`
- `KEYWORD`
- `NUMBER`
- `STRING`
- `OP`
- `PUNC`
- `COMMENT`

## 空白

空白（スペース / タブ / 改行）は読み飛ばされ，トークンは生成されません。

## 空白の扱い（補足）

- 字句解析は空白を無視します（トークン列には含まれません）。
- 行単位入力でも空白は同様に無視されます。

## 識別子

次の3形式を識別子として扱います。

1. 標準識別子  
   形式: `[{Unicode文字} _ {絵文字}][{Unicode文字}{Unicode数字}_ {絵文字}]*`  
   ※ Unicodeの文字判定は `str.isalpha/isalnum` 相当、絵文字は `unicodedata.category(ch) == "So"` を許可します。

2. `$` で始まる識別子  
   形式: `\$[{Unicode文字}{Unicode数字}{絵文字}]+`

3. クォート識別子  
   形式: `'[^']*'`  
   値にはクォートを含めます（例: `"'bar'"`）。

## キーワード

識別子はキーワードと一致する場合 `KEYWORD` トークンになります。  
識別子がキーワードで始まるが一致しない場合はエラーになります。  
例: キーワード `for` があるとき `for` は `KEYWORD`、`format` はエラーです。

標準キーワード:

- `break` `continue` `do` `else` `for` `if` `return` `while`

## 数値リテラル

無限長整数のみを扱います。

- 10進整数: `[0-9]+`
- 指数形式: `aaaEbb`（`aaa` と `bb` は `[0-9]+`）  
  値は `aaa * (10 ** bb)` として解釈します。  
  例: `12E3` → `12000`

## 文字列リテラル

ダブルクォートで囲まれた文字列を扱います。

- 形式: `"[^"]*"`
- 値はクォートを除いた中身です。

## コメント

行コメントは `//` から改行までです。  
コメントは `COMMENT` トークンとして出力されます。

## 演算子

演算子は定義リストに基づいて最長一致で切り出します。  
一致する定義が無い場合は1文字を `PUNC` として扱います。

## 句読点（PUNC）

演算子・識別子・数値・文字列・括弧対・キーワードに該当しない文字は、1文字単位で `PUNC` になります。

標準定義（字句解析器内で宣言）:

- `==` `!=` `>=` `<=` `&&` `||`
- `+=` `-=` `*=` `/=` `%=`
- `++` `--`
- `<-`
- `=` `+` `-` `*` `/` `%` `^` `<` `>`

演算子の定義は以下から結合されます。

1. トークナイザ内で宣言された標準定義
2. ソースと同名の `.mydef` 定義ファイル


## 定義ファイル（.mydef）

`.mydef` は Prolog 風のディレクティブ形式で記述します。

使用できるディレクティブ:

- `op('<lexeme>', <precedence>, <assoc>).`
- `pair('<open>', '<close>').`
- `comment('<prefix>').`
- `keyword('<name>').`

規則:

- `%` 以降はコメントとして無視されます（クォート内の `%` は除外）。
- トークナイザは `op/3` の `<lexeme>`，`pair/2`，`comment/1`，`keyword/1` を使用します。

## 位置情報

各トークンは位置情報を持ちます。

- `start`: `(index, line, column)`
- `end`: `(index, line, column)` （endは排他的）

`index` は0始まり，`line` と `column` は1始まりです。
`tokenize` / `tokenize_with_definitions` には `start_pos=(index, line, column)` を渡せます。

## 出力形式

出力はタプル形式で、先頭4要素は固定です。

- 先頭4要素: `(kind, value, start, end)`
- `start`/`end`: `(index, line, column)`（endは排他的）
- 必要に応じて第5要素に辞書（追加フィールド）を付与可能

## 解析途中の定義更新

`Tokenizer` は解析中に演算子・コメント開始記号・括弧対・キーワードを更新でき，その更新を取り消すこともできます。

利用可能な操作:

- `push_operator_update(add=None, remove=None)`
- `pop_operator_update()`
- `push_comment_update(add=None, remove=None)`
- `pop_comment_update()`
- `push_pair_update(add=None, remove=None)`
- `pop_pair_update()`
- `push_keyword_update(add=None, remove=None)`
- `pop_keyword_update()`

これらはスタック形式で管理され，`pop_*` は直前の更新を元に戻します。

引数の意味:
- `add`: 追加する要素のリスト
  - 演算子: 文字列（例: `"%%"`）
  - コメント開始記号: 文字列（例: `"//"`）
  - 括弧対: `(open, close)` のタプル（例: `("begin", "end")`）
  - キーワード: 文字列（例: `"for"`）
- `remove`: 削除する要素のリスト（現在の定義から一致するものを除外）

push/pop 実行例:

```python
t = Tokenizer()
t.push_operator_update(add=["%%"])
t.push_operator_update(add=["##"])
# %% と ## が有効
t.pop_operator_update()
# %% だけ有効
t.pop_operator_update()
# 標準状態に戻る
```

キーワード追加・削除の例:

```python
t = Tokenizer()
t.push_keyword_update(add=["class", "def"])
# class/def で始まる識別子はエラー
t.push_keyword_update(remove=["def"])
# class だけ残る
t.pop_keyword_update()
# class/def の状態に戻る
t.pop_keyword_update()
# 標準状態に戻る
```

## 実行例

サンプルコード:

```python
from MyNewTokenAnalyzer import tokenize_with_definitions

source = "for $x = 12E3 + \"hello\" // c\nx <- y"
tokens = tokenize_with_definitions(source, source_path="MyNewLang/example.mylang", start_pos=(0, 1, 1))
for t in tokens:
    print(t)
```

開始位置指定の例:

```python
from MyNewTokenAnalyzer import tokenize

tokens = tokenize("for x <- y", start_pos=(120, 10, 5))
for t in tokens:
    print(t)
```

想定される出力（抜粋）:

```
('KEYWORD', 'for', (0, 1, 1), (3, 1, 4))
('IDENT', '$x', (4, 1, 5), (6, 1, 7))
('OP', '=', (7, 1, 8), (8, 1, 9))
('NUMBER', 12000, (9, 1, 10), (13, 1, 14))
('STRING', 'hello', (16, 1, 17), (23, 1, 24))
('COMMENT', '// c', (24, 1, 25), (28, 1, 29))
```
