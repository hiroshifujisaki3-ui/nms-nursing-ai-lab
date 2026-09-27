# NMS 看護AI データ解析ラボ

看護学生向けの、コードを書かずに基本的なデータ解析を体験するための Streamlit アプリです。

## できること

- CSVの行数・列数・欠損値確認
- `DEATH_EVENT=0/1` の記述統計比較
- 年齢、駆出率、血清クレアチニン、血清Naの分布表示
- `time` と `source_row` を自動除外したロジスティック回帰
- Stratified K-fold cross-validation
- 混同行列、Accuracy、感度、特異度、Precision
- ROC曲線とAUC
- 分類しきい値の変更
- 授業用の考察問題と分析の限界
- 症例ごとの予測結果CSVの保存

## ローカルで実行

Python 3.11 などを用意して、このフォルダ内で以下を実行します。

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud で公開

1. GitHubに新しいリポジトリを作る
2. このフォルダの `app.py`, `requirements.txt`, `heart_failure_UCI_teaching_subset.csv` をアップロードする
3. Streamlit Community Cloud でGitHubリポジトリを指定する
4. Main file path に `app.py` を指定してDeployする
5. 発行されたURLを学生に配布する

学生側はPython、GitHub、Colabの操作は不要です。Webブラウザだけで利用できます。

## 授業上の注意

- `time` は説明変数には使用しません。
- `source_row` も識別用の列なので説明変数から除外します。
- モデル係数や群間差を因果関係として解釈しません。
- 授業用の小標本なので、性能値は不安定です。
- 教育目的であり、臨床判断には使用しません。
