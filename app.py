import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def setup_japanese_font():
    """グラフ中の日本語が文字化け（□）しないよう、使える日本語フォントを探して設定する。
    Streamlit Community Cloud では packages.txt の fonts-noto-cjk で Noto Sans CJK JP が入る。"""
    candidates = [
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", "MS Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_japanese_font()

st.set_page_config(
    page_title="NMS 看護AI データ解析ラボ",
    page_icon="🩺",
    layout="wide",
)

VARIABLE_INFO = {
    "source_row": "授業用サブセットで元データの行を示す番号（解析には使わない）",
    "age": "年齢（歳）",
    "anaemia": "貧血の有無（0/1）",
    "creatinine_phosphokinase": "血中CPK値",
    "diabetes": "糖尿病の有無（0/1）",
    "ejection_fraction": "駆出率（%）",
    "high_blood_pressure": "高血圧の有無（0/1）",
    "platelets": "血小板数",
    "serum_creatinine": "血清クレアチニン（mg/dL）",
    "serum_sodium": "血清Na（mEq/L）",
    "sex": "性別（原データでは0/1）",
    "smoking": "喫煙の有無（0/1）",
    "time": "追跡期間（日）。この授業では説明変数に使わない",
    "DEATH_EVENT": "追跡期間中の死亡イベント（0=なし, 1=あり）",
}

JP_LABEL = {
    "age": "年齢（歳）",
    "ejection_fraction": "駆出率（%）",
    "serum_creatinine": "血清クレアチニン（mg/dL）",
    "serum_sodium": "血清Na（mEq/L）",
    "creatinine_phosphokinase": "CPK",
    "platelets": "血小板数",
    "time": "追跡期間（日）",
}
GROUP_LABEL = {0: "生存（0）", 1: "死亡イベント（1）"}
GROUP_COLOR = {0: "#1f77b4", 1: "#ff7f0e"}


def label(var):
    return f"{var}\n{JP_LABEL[var]}" if var in JP_LABEL else var


FOCUS_VARS = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium"]
EXCLUDE_ALWAYS = ["source_row", "time"]
TARGET = "DEATH_EVENT"

st.title("🩺 NMS 看護AI データ解析ラボ")
st.caption("Heart Failure Clinical Records：記述統計 → 可視化 → ロジスティック回帰 → 混同行列・ROC → time実験")

with st.expander("このアプリで学ぶこと", expanded=True):
    st.markdown(
        """
- CSVデータの行数・欠損・変数を確認する
- `DEATH_EVENT=0` と `1` の分布を比較する
- `time` を使わずにロジスティック回帰を行う
- Accuracyだけでなく、感度・特異度・混同行列を読む
- ROC曲線とAUCの意味を考える
- `time`（未来の情報）を入れると何が起きるかを確かめる
- **関連を因果関係と断定しない**
        """
    )

st.warning("授業上の重要事項：`time`（追跡期間＝未来の情報）は、初期設定ではモデルの説明変数から外しています。`source_row` は識別用の番号なので常に除外します。")

# --------------------
# Data loading
# --------------------
st.header("1. CSVを読み込む")
uploaded = st.file_uploader("授業用CSVを選んでください", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    data_source = "アップロードしたCSV"
else:
    try:
        df = pd.read_csv("heart_failure_UCI_teaching_subset.csv")
        data_source = "アプリに同梱した授業用サンプルCSV"
        st.info("CSVをまだ選んでいないため、同梱サンプルを表示しています。")
    except FileNotFoundError:
        st.stop()

required = {TARGET, "time"}
if not required.issubset(df.columns):
    st.error(f"必要な列 {sorted(required)} が見つかりません。列名を確認してください。")
    st.stop()

if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
    st.error("DEATH_EVENT は 0/1 の二値である必要があります。")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("行数", len(df))
c2.metric("列数", len(df.columns))
c3.metric("欠損セル数", int(df.isna().sum().sum()))
c4.metric("DEATH_EVENT=1", int((df[TARGET] == 1).sum()))
st.caption(f"使用データ：{data_source}")

with st.expander("データの先頭を見る"):
    st.dataframe(df.head(10), width="stretch")

with st.expander("各列の意味と欠損を確認", expanded=True):
    info = pd.DataFrame({
        "列名": df.columns,
        "意味": [VARIABLE_INFO.get(c, "ユーザーが追加した列") for c in df.columns],
        "欠損数": [int(df[c].isna().sum()) for c in df.columns],
        "データ型": [str(df[c].dtype) for c in df.columns],
    })
    st.dataframe(info, width="stretch", hide_index=True)

# --------------------
# Descriptive comparison
# --------------------
st.header("2. DEATH_EVENT=0 と 1 を比べる")
available_focus = [v for v in FOCUS_VARS if v in df.columns]
numeric_cols = [
    c for c in df.columns
    if c not in ["source_row", TARGET] and pd.api.types.is_numeric_dtype(df[c])
]

if available_focus:
    # 2-1 summary table
    st.subheader("2-1. 要約統計の表")
    summary = df.groupby(TARGET)[available_focus].agg(["count", "mean", "median"]).T
    summary.columns = [f"DEATH_EVENT={c}" for c in summary.columns]
    st.dataframe(summary.round(3), width="stretch")

    # 2-2 distribution
    st.subheader("2-2. 分布を見る（箱ひげ図・バイオリン図）")
    d1, d2 = st.columns([2, 1])
    with d1:
        plot_vars = st.multiselect(
            "分布を見る変数（2つまでがおすすめ）",
            available_focus,
            default=available_focus[:2],
            max_selections=4,
        )
    with d2:
        plot_kind = st.radio("図の種類", ["箱ひげ図", "バイオリン図"], horizontal=True)
        show_points = st.checkbox("1人ずつの値（点）も表示", value=True)

    if plot_vars:
        cols = st.columns(min(2, len(plot_vars)))
        rng = np.random.default_rng(0)
        for i, var in enumerate(plot_vars):
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(5, 3.8))
                vals = [df.loc[df[TARGET] == g, var].dropna().values for g in [0, 1]]
                if plot_kind == "箱ひげ図":
                    bp = ax.boxplot(vals, positions=[0, 1], widths=0.5, patch_artist=True,
                                    showfliers=not show_points)
                    for patch, g in zip(bp["boxes"], [0, 1]):
                        patch.set_facecolor(GROUP_COLOR[g])
                        patch.set_alpha(0.35)
                    for med in bp["medians"]:
                        med.set_color("black")
                        med.set_linewidth(2)
                else:
                    parts = ax.violinplot(vals, positions=[0, 1], showmeans=False,
                                          showmedians=True, showextrema=True)
                    for body, g in zip(parts["bodies"], [0, 1]):
                        body.set_facecolor(GROUP_COLOR[g])
                        body.set_alpha(0.35)
                if show_points:
                    for g, v in zip([0, 1], vals):
                        ax.scatter(g + rng.uniform(-0.12, 0.12, len(v)), v, s=14,
                                   color=GROUP_COLOR[g], alpha=0.8, zorder=3)
                ax.set_xticks([0, 1], [f"{GROUP_LABEL[g]}\nn={len(v)}" for g, v in zip([0, 1], vals)])
                ax.set_xlabel("DEATH_EVENT")
                ax.set_ylabel(label(var))
                ax.set_title(f"{var} の分布（{plot_kind}）")
                ax.grid(alpha=0.2)
                st.pyplot(fig, clear_figure=True)
        st.caption("箱ひげ図：箱の中の太線＝中央値、箱＝真ん中50%の範囲。点＝1人ずつの値。")

    # 2-3 mean comparison
    st.subheader("2-3. 2群の平均を並べる")
    mean_vars = st.multiselect(
        "平均を比べる変数",
        [c for c in numeric_cols if c != "time"],
        default=available_focus,
        key="mean_vars",
    )
    if mean_vars:
        overall = df[mean_vars].mean()
        g_mean = df.groupby(TARGET)[mean_vars].mean()
        ratio = g_mean.div(overall) * 100
        fig, ax = plt.subplots(figsize=(8, 3.8))
        x = np.arange(len(mean_vars))
        w = 0.38
        for k, g in enumerate([0, 1]):
            bars = ax.bar(x + (k - 0.5) * w, ratio.loc[g].values, w,
                          color=GROUP_COLOR[g], label=GROUP_LABEL[g])
            for bx, val in zip(bars, g_mean.loc[g].values):
                ax.text(bx.get_x() + bx.get_width() / 2, bx.get_height() + 1,
                        (f"{val:.2f}" if val < 10 else f"{val:.1f}" if val < 1000 else f"{val:.0f}"), ha="center", va="bottom", fontsize=9)
        ax.axhline(100, color="gray", linewidth=1, linestyle="--")
        ax.set_xticks(x, [label(v) for v in mean_vars], fontsize=9)
        ax.set_ylabel("全体平均を100としたときの値")
        ax.set_title("2群の平均（棒の上の数字＝実際の平均値）")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, max(130, ratio.values.max() * 1.15))
        st.pyplot(fig, clear_figure=True)
        mean_tbl = g_mean.T.round(2)
        mean_tbl.columns = [GROUP_LABEL[c] for c in mean_tbl.columns]
        mean_tbl["差（1 − 0）"] = (g_mean.loc[1] - g_mean.loc[0]).round(2).values
        st.dataframe(mean_tbl, width="stretch")
        st.caption("単位が違う変数を同じ図で比べるため、全体平均を100として表示しています。差が大きい変数は『候補』であり、原因ではありません。")

    # 2-4 scatter
    st.subheader("2-4. 2つの変数を同時に見る（散布図）")
    s1, s2 = st.columns(2)
    scatter_choices = [c for c in numeric_cols if c != "time"]
    with s1:
        x_var = st.selectbox("横軸", scatter_choices,
                             index=scatter_choices.index("ejection_fraction") if "ejection_fraction" in scatter_choices else 0)
    with s2:
        y_var = st.selectbox("縦軸", scatter_choices,
                             index=scatter_choices.index("serum_creatinine") if "serum_creatinine" in scatter_choices else min(1, len(scatter_choices) - 1))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for g, mk in [(0, "o"), (1, "^")]:
        sub = df[df[TARGET] == g]
        ax.scatter(sub[x_var], sub[y_var], s=36, marker=mk, color=GROUP_COLOR[g],
                   alpha=0.8, label=f"{GROUP_LABEL[g]}  n={len(sub)}")
    ax.set_xlabel(label(x_var).replace("\n", " "))
    ax.set_ylabel(label(y_var).replace("\n", " "))
    ax.set_title(f"{x_var} × {y_var}（色と形＝DEATH_EVENT）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    st.pyplot(fig, clear_figure=True)
    st.caption("2群が1本の線できれいに分かれるかを見てみましょう。分かれないなら、1つの値だけで個人を判定できないことを意味します。")

st.info("平均値や分布の違いは『関連』の手がかりです。ここから因果関係を断定することはできません。")

# --------------------
# Logistic regression
# --------------------
st.header("3. ロジスティック回帰")

candidate_features = [
    c for c in df.columns
    if c not in ["source_row", TARGET]
    and pd.api.types.is_numeric_dtype(df[c])
]
default_features = [c for c in candidate_features if c != "time"]

selected_features = st.multiselect(
    "説明変数を選ぶ",
    candidate_features,
    default=default_features,
)

st.caption("`source_row` は選択肢に表示されません。`time` は選べますが、初期状態では外しています。")
if "time" in selected_features:
    st.error(
        "`time`（その後何日追跡したか）が説明変数に入っています。"
        "time は初診時にはまだ分からない**未来の情報**なので、初診時の予測モデルとしては不適切です。"
        "AUCが上がっても「ずるい高得点」であることに注意してください（5. の time実験も参照）。"
    )

threshold = st.slider("分類のしきい値", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

def make_model(X, features):
    binary_cols, continuous_cols = [], []
    for c in features:
        vals = set(X[c].dropna().unique().tolist())
        if vals.issubset({0, 1}) and len(vals) <= 2:
            binary_cols.append(c)
        else:
            continuous_cols.append(c)
    transformers = []
    if continuous_cols:
        transformers.append(("continuous", StandardScaler(), continuous_cols))
    if binary_cols:
        transformers.append(("binary", "passthrough", binary_cols))
    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers)),
        ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
    ])
    return model, continuous_cols, binary_cols


if st.button("ロジスティック回帰を実行", type="primary", width="stretch"):
    st.session_state["model_run"] = True
run = st.session_state.get("model_run", False)
if run:
    st.caption("結果を表示中です。しきい値や説明変数を変えると、結果は自動で更新されます。")

if run:
    if len(selected_features) == 0:
        st.error("説明変数を1つ以上選んでください。")
        st.stop()

    model_df = df[selected_features + [TARGET]].dropna().copy()
    X = model_df[selected_features]
    y = model_df[TARGET].astype(int)

    counts = y.value_counts()
    if len(counts) < 2:
        st.error("DEATH_EVENT に0と1の両方が必要です。")
        st.stop()

    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        st.error("各クラスに少なくとも2例必要です。")
        st.stop()

    model, continuous_cols, binary_cols = make_model(X, selected_features)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= threshold).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, pred)
    sens = recall_score(y, pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = precision_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, prob)

    st.subheader("評価結果")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("感度", f"{sens:.3f}")
    m3.metric("特異度", f"{spec:.3f}")
    m4.metric("Precision", f"{prec:.3f}")
    m5.metric("ROC-AUC", f"{auc:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("混同行列")
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        im = ax.imshow(cm)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
        ax.set_xticks([0, 1], ["予測 0", "予測 1"])
        ax.set_yticks([0, 1], ["実際 0", "実際 1"])
        ax.set_xlabel("予測")
        ax.set_ylabel("実際")
        ax.set_title(f"しきい値 = {threshold:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)
        st.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    with right:
        st.subheader("ROC曲線")
        fpr, tpr, thresholds = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", label="ランダム")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("偽陽性率 (1 - 特異度)")
        ax.set_ylabel("真陽性率 (感度)")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)

    st.subheader("結果を読む")
    st.markdown(
        f"""
- Accuracy は **{acc:.3f}** です。
- 感度は **{sens:.3f}** で、実際に `DEATH_EVENT=1` の人のうちどれだけを1と判定できたかを表します。
- 特異度は **{spec:.3f}** で、実際に `DEATH_EVENT=0` の人のうちどれだけを0と判定できたかを表します。
- ROC-AUC は **{auc:.3f}** です。
- これらは**この小さな授業用データに対する予測性能**であり、臨床でそのまま使える性能を意味しません。
        """
    )

    # Fit whole data only to show coefficients as descriptive model parameters.
    model.fit(X, y)
    feature_names = continuous_cols + binary_cols
    coef = model.named_steps["logreg"].coef_[0]
    coef_df = pd.DataFrame({"変数": feature_names, "係数": coef, "exp(係数)": np.exp(coef)})
    coef_df["|係数|"] = coef_df["係数"].abs()
    coef_df = coef_df.sort_values("|係数|", ascending=False).drop(columns="|係数|")

    with st.expander("参考：全データでfitした係数を見る"):
        st.dataframe(coef_df.round(3), width="stretch", hide_index=True)
        st.caption("連続変数は標準化されています。係数は因果効果ではありません。")

    pred_df = model_df[[TARGET]].copy()
    pred_df["予測確率"] = prob
    pred_df["予測クラス"] = pred
    csv_bytes = pred_df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "各症例の予測結果をCSVで保存",
        data=csv_bytes,
        file_name="heart_failure_predictions.csv",
        mime="text/csv",
    )

    st.header("4. 考えてみよう")
    st.markdown(
        f"""
1. しきい値を **{threshold:.2f}** から下げると、感度と特異度はどう変わりそうですか？
2. 医療・看護で **False Negative（見逃し）** と **False Positive（過剰な警告）** は、それぞれどんな問題を起こし得ますか？
3. Accuracy が高ければ、医療AIとして十分でしょうか？
4. 年齢や血清クレアチニンなどと `DEATH_EVENT` の関連が見えても、なぜ因果関係とは言えないのでしょうか？
        """
    )

    st.subheader("この分析の限界")
    st.markdown(
        """
1. **標本数が少ない**：授業用サブセットなので、データの分け方によって評価値が変動しやすいです。
2. **交絡や因果を扱っていない**：ロジスティック回帰の係数は関連を表すもので、原因を証明するものではありません。
3. **生存時間解析ではない**：この授業では `time` を説明変数に使っておらず、追跡期間の違いを扱うKaplan–Meier法やCox回帰とは目的が異なります。
        """
    )

    with st.expander("このアプリが行っている解析コード（学習用）"):
        st.code(
            """# 概念的には次の処理をしています
X = df[selected_features]       # source_row は除外。time は初期設定では除外
y = df[\"DEATH_EVENT\"]

# 連続変数を標準化
# ロジスティック回帰
# Stratified K-fold cross-validation
# 各症例について学習に使っていないモデルで予測確率を計算
# しきい値で0/1分類
# 混同行列、感度、特異度、ROC-AUCを計算
""",
            language="python",
        )

# --------------------
# Leakage experiment
# --------------------
st.header("5. 発展：time を入れると何が起きる？（データリーケージ実験）")
st.markdown(
    """
`time` は「その後何日追跡したか」＝**初診時にはまだ分からない未来の情報**です。
ここでは、**同じデータ・同じ5分割交差検証**で、説明変数の組み合わせだけを変えた3つのモデルのAUCを比べます。
"""
)
base_features = [
    c for c in df.columns
    if c not in EXCLUDE_ALWAYS + [TARGET] and pd.api.types.is_numeric_dtype(df[c])
]
leak_models = {
    "A：基本モデル（time以外の全変数）": base_features,
    "B：A ＋ time（未来の情報を混ぜる）": base_features + ["time"],
    "C：4変数モデル（age・EF・Cr・Na）": [v for v in FOCUS_VARS if v in df.columns],
}
if st.button("3つのモデルを比べる", width="stretch"):
    st.session_state["leak_run"] = True
if st.session_state.get("leak_run", False):
    y_all = df[TARGET].astype(int)
    rows = []
    for name, feats in leak_models.items():
        sub = df[feats + [TARGET]].dropna()
        Xs, ys = sub[feats], sub[TARGET].astype(int)
        k = min(5, int(ys.value_counts().min()))
        m, _, _ = make_model(Xs, feats)
        p = cross_val_predict(m, Xs, ys, cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=42),
                              method="predict_proba")[:, 1]
        rows.append({"モデル": name, "変数の数": len(feats), "ROC-AUC": roc_auc_score(ys, p),
                     "Accuracy（しきい値0.5）": accuracy_score(ys, (p >= 0.5).astype(int)),
                     "説明変数": ", ".join(feats)})
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    bars = ax.bar(["A：基本", "B：A＋time", "C：4変数"], res["ROC-AUC"], color=colors)
    for bx, v in zip(bars, res["ROC-AUC"]):
        ax.text(bx.get_x() + bx.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC（5分割交差検証）")
    ax.set_title("説明変数の組み合わせだけを変えたときのAUC")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig, clear_figure=True)
    st.dataframe(res.round(3), width="stretch", hide_index=True)
    st.error("B のAUCが高く見えても、初診時には分からない `time` を使っているため、初診時の予測モデルとしては使えません。")
    st.caption("B は教材としての比較のために time を入れています。3. のロジスティック回帰でも time を選べますが、初期設定では外しています。")

st.divider()
st.caption("教育目的のアプリです。診断・治療判断には使用しないでください。")
