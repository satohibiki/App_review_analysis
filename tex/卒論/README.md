# 卒論テンプレート (platex, jsbook, 単体版)
jsbook を用いた新世代の卒論テンプレート

## 特徴
### タイプセットが早い
cover, abstract, thesis と別々にコンパイルするより無駄が無い.
### latexmkのサポート
makeコマンド以外のタイプセット補助を使うことが可能に. 
### 一つのtexファイルだけ
設定が散逸しない.

## 簡単な使用方法
1. thesis.tex に自分の情報を入れる.
2. 自分の卒論を texファイル で書く.
3. abstractContents.tex に論文要旨を入れる.
4. thesis.tex に \input{texファイルのパス} で書いた texファイル を挿入する.
5. make

## テンプレートの構成について
- thesis.tex: 卒論本体
- thesis.bib: 卒論の参考文献情報ファイル
- thesis.pdf: 出来上がった卒論(表紙, 論文要旨付き)
- abstractContents.tex: 論文要旨の本文
- (ipa/ms).map: (ipa|ms)フォントを埋め込む際のmapファイル
- Makefile: makeする時に用いる設定ファイル
- .latexmkrc: latexmkでタイプセットする場合に使う設定ファイル
- (clean, make_me, latexmk).bat: windowsでダブルクリックで使う際のファイル
- (cover|abstract|thesis).sty: 卒論用パッケージ
- pxjahyper.sty: pyperref の日本語対応用パッケージ
- jlisting.sty: listings の日本語対応用パッケージ
- images/: 画像ファイルを入れておくディレクトリ
- sample/: サンプルの卒論文章(tex)が入ったディレクトリ

## luatex版との相違
- bookmark.sty が使用不可
- pdfcomment.sty が使用不可
- lua によるマクロが記述出来ない
