import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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


# ============================================================
# Japanese font setup for Streamlit Community Cloud
# ============================================================
def setup_japanese_font():
    """Find and register a Japanese font. Return (font_properties, font_name, path)."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
    ]

    # Also search common Linux font directories dynamically.
    candidates += glob.glob("/usr/share/fonts/**/*NotoSans*CJK*Regular*", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/*NotoSans*JP*Regular*", recursive=True)
    candidates += glob.glob("/usr/local/share/fonts/**/*NotoSans*CJK*Regular*", recursive=True)
    candidates += glob.glob("/usr/local/share/fonts/**/*NotoSans*JP*Regular*", recursive=True)

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    for path in candidates:
        if os.path.isfile(path):
            try:
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                family = prop.get_name()
                plt.rcParams["font.family"] = family
                plt.rcParams["axes.unicode_minus"] = False
                return prop, family, path
            except Exception:
                pass

    # Last attempt: search already installed font families.
    for family in [
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
    ]:
        try:
            path = fm.findfont(family, fallback_to_default=False)
            if path and os.path.isfile(path):
                prop = fm.FontProperties(fname=path)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return prop, prop.get_name(), path
        except Exception:
            pass

    plt.rcParams["axes.unicode_minus"] = False
    return None, None, None


JP_FONT, JP_FONT_NAME, JP_FONT_PATH = setup_japanese_font()
HAS_JAPANESE_FONT = JP_FONT is not None


def jp_or_en(japanese, english):
    """Use Japanese on plots only when a Japanese font is available."""
    return japanese if HAS_JAPANESE_FONT else english


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

FOCUS_VARS = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium"]
EXCLUDE_ALWAYS = ["source_row", "time"]
TARGET = "DEATH_EVENT"

st.title("🩺 NMS 看護AI データ解析ラボ")
st.caption("Heart Failure Clinical Records：記述統計 → ロジスティック回帰 → 混同行列 → ROC")

if HAS_JAPANESE_FONT:
    st.success(f"グラフ用日本語フォントを認識しました：{JP_FONT_NAME}", icon="✅")
else:
    st.warning(
        "日本語フォントが見つからなかったため、グラフ内の文字だけ英語表示にします。"
        " Streamlit本文は日本語のままです。",
        icon="⚠️",
    )

with st.expander("フォント診断情報"):
    if HAS_JAPANESE_FONT:
        st.code(f"Font name: {JP_FONT_NAME}\nFont path: {JP_FONT_PATH}")
    else:
        st.code(
            "Japanese font not found.\n"
            "GitHub の app.py と同じ階層に packages.txt を置き、\n"
            "中身を fonts-noto-cjk としてください。"
        )

with st.expander("このアプリで学ぶこと", expanded=True):
    st.markdown(
        """
- CSVデータの行数・欠損・変数を確認する
- `DEATH_EVENT=0` と `1` の分布を比較する
- `time` を使わずにロジスティック回帰を行う
- Accuracyだけでなく、感度・特異度・混同行列を読む
- ROC曲線とAUCの意味を考える
- **関連を因果関係と断定しない**
        """
    )

st.warning("授業上の重要事項：`time` と `source_row` はモデルの説明変数から自動的に除外します。")

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
        st.error("heart_failure_UCI_teaching_subset.csv が見つかりません。CSVをアップロードしてください。")
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
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("各列の意味と欠損を確認", expanded=True):
    info = pd.DataFrame({
        "列名": df.columns,
        "意味": [VARIABLE_INFO.get(c, "ユーザーが追加した列") for c in df.columns],
        "欠損数": [int(df[c].isna().sum()) for c in df.columns],
        "データ型": [str(df[c].dtype) for c in df.columns],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

# --------------------
# Descriptive comparison
# --------------------
st.header("2. DEATH_EVENT=0 と 1 を比べる")
available_focus = [v for v in FOCUS_VARS if v in df.columns]
if available_focus:
    summary_rows = []
    for var in available_focus:
        for g in [0, 1]:
            s = df.loc[df[TARGET] == g, var].dropna()
            summary_rows.append({
                "変数": var,
                "DEATH_EVENT": g,
                "n": len(s),
                "平均": s.mean(),
                "中央値": s.median(),
            })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.round(3), use_container_width=True, hide_index=True)

    plot_vars = st.multiselect(
        "分布を見る変数（2つまでがおすすめ）",
        available_focus,
        default=available_focus[:2],
        max_selections=4,
    )

    if plot_vars:
        cols = st.columns(min(2, len(plot_vars)))
        for i, var in enumerate(plot_vars):
            with cols[i % len(cols)]:
                fig, ax = plt.subplots(figsize=(5, 3.6))
                vals = [
                    df.loc[df[TARGET] == g, var].dropna().values
                    for g in [0, 1]
                ]
                ax.violinplot(
                    vals,
                    positions=[0, 1],
                    showmeans=False,
                    showmedians=True,
                    showextrema=True,
                )
                ax.set_xticks([0, 1], ["0", "1"])
                ax.set_xlabel("DEATH_EVENT")
                ax.set_ylabel(var)
                ax.set_title(jp_or_en(f"{var} の分布", f"Distribution of {var}"))
                ax.grid(alpha=0.2)
                st.pyplot(fig, clear_figure=True)

st.info("平均値や分布の違いは『関連』の手がかりです。ここから因果関係を断定することはできません。")

# --------------------
# Logistic regression
# --------------------
st.header("3. ロジスティック回帰")

candidate_features = [
    c for c in df.columns
    if c not in EXCLUDE_ALWAYS + [TARGET]
    and pd.api.types.is_numeric_dtype(df[c])
]

selected_features = st.multiselect(
    "説明変数を選ぶ",
    candidate_features,
    default=candidate_features,
)

st.caption("`time` と `source_row` は選択肢に表示されません。")
threshold = st.slider("分類のしきい値", 0.10, 0.90, 0.50, 0.05)
run = st.button("ロジスティック回帰を実行", type="primary", use_container_width=True)

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

    binary_cols = []
    continuous_cols = []
    for c in selected_features:
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

    preprocess = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("preprocess", preprocess),
        ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
    ])

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

        ax.set_xticks(
            [0, 1],
            [
                jp_or_en("予測 0", "Predicted 0"),
                jp_or_en("予測 1", "Predicted 1"),
            ],
        )
        ax.set_yticks(
            [0, 1],
            [
                jp_or_en("実際 0", "Actual 0"),
                jp_or_en("実際 1", "Actual 1"),
            ],
        )
        ax.set_xlabel(jp_or_en("予測", "Predicted"))
        ax.set_ylabel(jp_or_en("実際", "Actual"))
        ax.set_title(jp_or_en(f"しきい値 = {threshold:.2f}", f"Threshold = {threshold:.2f}"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)
        st.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    with right:
        st.subheader("ROC曲線")
        fpr, tpr, thresholds = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label=jp_or_en("ランダム", "Random"),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(jp_or_en("偽陽性率（1 - 特異度）", "False positive rate (1 - specificity)"))
        ax.set_ylabel(jp_or_en("真陽性率（感度）", "True positive rate (sensitivity)"))
        ax.set_title(jp_or_en("ROC曲線", "ROC curve"))
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
        st.dataframe(coef_df.round(3), use_container_width=True, hide_index=True)
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
            '''# 概念的には次の処理をしています
X = df[selected_features]       # time と source_row は除外
y = df["DEATH_EVENT"]

# 連続変数を標準化
# ロジスティック回帰
# Stratified K-fold cross-validation
# 各症例について学習に使っていないモデルで予測確率を計算
# しきい値で0/1分類
# 混同行列、感度、特異度、ROC-AUCを計算
''',
            language="python",
        )

st.divider()
st.caption("教育目的のアプリです。診断・治療判断には使用しないでください。")
