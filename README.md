# NMS 看護AI データ解析ラボ

看護学生向けの、コードを書かずに基本的なデータ解析を体験するための Streamlit アプリです。

## できること

- CSVの行数・列数・欠損値確認
- `DEATH_EVENT=0/1` の記述統計比較
- 年齢、駆出率、血清クレアチニン、血清Naの分布表示（箱ひげ図／バイオリン図、1人ずつの点の表示）
- 2群の平均を「全体平均=100」でそろえた棒グラフ
- 2つの変数を選んで描く散布図（DEATH_EVENTで色分け）
- ロジスティック回帰（`source_row` は常に除外。`time` は選択可能だが初期設定では除外し、選ぶと警告を表示）
- Stratified K-fold cross-validation
- 混同行列、Accuracy、感度、特異度、Precision
- ROC曲線とAUC
- 分類しきい値の変更
- 授業用の考察問題と分析の限界
- 症例ごとの予測結果CSVの保存
- 発展：`time` を入れたモデルとのAUC比較（データリーケージ実験）
- グラフの日本語表示（`packages.txt` で日本語フォントを導入）

## ローカルで実行

Python 3.11 などを用意して、このフォルダ内で以下を実行します。

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud で公開

1. GitHubに新しいリポジトリを作る
2. このフォルダの `app.py`, `requirements.txt`, `packages.txt`, `heart_failure_UCI_teaching_subset.csv` をアップロードする
3. Streamlit Community Cloud でGitHubリポジトリを指定する
4. Main file path に `app.py` を指定してDeployする
5. 発行されたURLを学生に配布する

学生側はPython、GitHub、Colabの操作は不要です。Webブラウザだけで利用できます。

## 授業上の注意

- `time` は初期設定では説明変数に使用しません。学生が選ぶことはできますが、その場合は「未来の情報」である旨の警告が出ます。
- `source_row` も識別用の列なので説明変数から除外します。
- モデル係数や群間差を因果関係として解釈しません。
- 授業用の小標本なので、性能値は不安定です。
- 教育目的であり、臨床判断には使用しません。
