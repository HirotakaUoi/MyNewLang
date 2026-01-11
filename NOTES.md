# 引き継ぎメモ

## 概要

- プロジェクト: MyNewLang
- 主な構成: 字句解析器 / パーサー / ドキュメント

## 現在の仕様（要点）

- 字句解析: `MyNewTokenAnalyzer.py` / `MyNewTokenAnalyzer_Functional.py`
  - 識別子がキーワードで始まる場合はエラー
  - 未知文字は1文字単位で `PUNC`
  - `tokenize_line()` で行単位の字句解析が可能
- パーサー: `MyNewTestPaser.py`
  - ASTは内部辞書、出力はタプルの木表示
  - ブロックは `{ ... }` のみ（`pair(...)` はブロックにしない）
  - 対話モードとバッチモードを切替可能 (`:batch` / `:interactive`)

## 主な使い方

- バッチ解析:
  - `batch_parse_text(source)`
  - `batch_parse_file(path)`
- 対話解析:
  - `python3 MyNewLang/MyNewTestPaser.py`

## 直近の変更

- ブロックを `{}` のみに限定
- パーサー仕様を `MyNewTestPaser.md` に追加
- READMEを字句解析/パーサーで分割

## 未解決・今後の検討

- 対話モードのエラー表示の改善
- 行入力バッファの永続化（必要なら）
