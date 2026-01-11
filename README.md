# MyNewLang README

このフォルダには，新しい言語向けの字句解析器が含まれます。

識別子は Unicode 文字と絵文字（`unicodedata.category == "So"`）を扱えます。
識別子がキーワードと一致する場合は `KEYWORD` トークンになります。
出力は `(kind, value, start, end)` のタプル形式です（必要なら第5要素に辞書で拡張）。`start`/`end` は `(index, line, column)` のタプルです。
`start_pos=(index, line, column)` を渡すと、その位置から始まるものとして位置情報が計算されます。
キーワードで始まるが一致しない識別子はエラーになります。
演算子・識別子・数値・文字列・括弧対・キーワードに該当しない文字は `PUNC` になります。
空白（スペース / タブ / 改行）は字句解析で無視され、トークン列には含まれません。  空白判定は `str.isspace()` に従い、全角スペースや各種Unicodeの空白も対象です。


標準キーワード:

- `break` `continue` `do` `else` `for` `if` `return` `while`

## 字句解析

### 使い方（簡単）

```python
from MyNewTokenAnalyzer import tokenize_with_definitions

tokens = tokenize_with_definitions(
    "x <- 12E3 + \"hello\"",
    source_path="MyNewLang/example.mylang",
    start_pos=(0, 1, 1),
)
```

インタプリタ向けの開始位置指定例:

```python
from MyNewTokenAnalyzer import tokenize

tokens = tokenize(
    "for x <- y",
    start_pos=(120, 10, 5),
)
```

## パーサー

文法仕様は `MyNewTestPaser.md` を参照してください。

### 使い方（バッチ/対話）

### バッチモード

テキストをまとめて解析する関数:

```python
from MyNewTestPaser import batch_parse_text

source = "program test { var a; a = 1; }"
batch_parse_text(source)
```

ファイルパスを渡す関数:

```python
from MyNewTestPaser import batch_parse_file

batch_parse_file("path/to/example.mylang")
```

### 対話モード

```
python3 MyNewLang/MyNewTestPaser.py
```

対話中のコマンド:
- `:batch` バッチモードに切り替え
- `:interactive` 対話モードに戻す

履歴:
- 実行ファイル名と同名の `.history` に保存されます（最大100件）

## 定義ファイル形式（.mydef）

定義は Prolog 風のディレクティブで記述し，ソースと同じベース名の
`.mydef` ファイルとして保存します。

例（`example.mydef`）:

```
% 演算子 / 括弧対 / コメントの定義
op('%%', 40, xfy).
op('##', 40, xfy).
pair('begin','end').
comment('//').
keyword('loop').
```

字句解析器は `op/3`, `pair/2`, `comment/1` のみを参照します。

## 解析途中での更新

解析中に演算子・コメント開始記号・括弧対を更新し，後で取り消すことができます。

```python
from MyNewTokenAnalyzer import Tokenizer

t = Tokenizer()
t.push_operator_update(add=["%%"])
t.push_comment_update(add=["#"])
t.push_pair_update(add=[("begin", "end")])
t.push_keyword_update(add=["for"])
# ...解析...
t.pop_keyword_update()
t.pop_pair_update()
t.pop_comment_update()
t.pop_operator_update()
```

引数:
- `add`: 追加する要素のリスト（演算子は文字列，コメントは接頭辞文字列，括弧対は `(open, close)` のタプル）
- `remove`: 削除する要素のリスト（現在の定義から一致するものを除外）

push/pop 実行例:

```python
t = Tokenizer()
t.push_operator_update(add=["%%"])
t.push_operator_update(add=["##"])
# %% と ## が有効
t.pop_operator_update()
# %% だけ有効（最後の更新を取り消し）
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
