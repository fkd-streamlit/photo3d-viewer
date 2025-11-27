import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import plotly.graph_objects as go
import tempfile
import os
import uuid

# ------------------------------------------------------
# Streamlit 基本設定
# ------------------------------------------------------
st.set_page_config(page_title="Photo3D Viewer", layout="wide")
st.title("📸 Photo3D Viewer – 写真が立体に変わる瞬間")

st.write(
    "1枚の写真から **AI が奥行きを推定して 3D 点群として可視化** します。"
)
st.markdown("""
### 🌐 Photo3D Viewer へようこそ  
このアプリは、1枚の写真からAIを使って **奥行き(Depth) を推定し、3D点群として可視化** します。

#### 🔧 推奨画像
- 解像度：横 2000px 以下がおすすめ  
- 明暗の差がある画像は奥行き推定が安定します  

#### 🎛 奥行き調整のコツ
- **奥行き強調倍率** を上げると立体感が増します  
- **点群密度** を上げると詳細になりますが重くなります  

#### 📌 使い方（3ステップ）
1. 左側のサイドバーから画像をアップロード  
2. 深度推定が自動実行 → 深度マップが表示されます  
3. スライダーで奥行き/密度を調整して 3Dビューをお楽しみください  
""")


# ------------------------------------------------------
# 深度推定モデル（Depth Anything）をキャッシュして読み込み
# ------------------------------------------------------
@st.cache_resource
def load_depth_model():
    # Hugging Face の Depth Anything モデル
    return pipeline(
        "depth-estimation",
        model="LiheYoung/depth-anything-base-hf",
    )


# ------------------------------------------------------
# 画像アップロード UI（サイドバー）
# ------------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "画像をアップロード", type=["jpg", "jpeg", "png"]
)

# 点群密度と奥行きパラメータ（サイドバー）
step = st.sidebar.slider("点群密度（間引きピクセル数）", 2, 20, 8)
exp_factor = st.sidebar.slider("奥行き強調（指数）", 1.0, 3.0, 1.6, 0.1)
z_scale = st.sidebar.slider("奥行きスケール係数", 50, 2000, 600)
max_points = st.sidebar.number_input(
    "最大点数（データ量制限）", min_value=20000, max_value=200000, value=120000, step=10000
)

st.sidebar.markdown("---")
st.sidebar.markdown("※ 点数を増やすとキレイになりますが、重くなります。")

if not uploaded_file:
    st.info("左のサイドバーから画像をアップロードしてください。")
else:
    # --------------------------------------------------
    # 一時フォルダに保存（日本語ファイル名対策）
    # --------------------------------------------------
    temp_dir = tempfile.mkdtemp()
    safe_name = f"{uuid.uuid4().hex}.png"
    img_path = os.path.join(temp_dir, safe_name)

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # OpenCV で読み込み → RGB 変換
    img = cv2.imread(img_path)
    if img is None:
        st.error("画像の読み込みに失敗しました。別の画像を試してください。")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    h, w = img_rgb.shape[:2]

    col1, col2 = st.columns(2)
    col1.image(img_rgb, caption="入力画像", width="stretch")

    # --------------------------------------------------
    # Depth Anything による深度推定
    # --------------------------------------------------
    with st.spinner("深度推定モデルをロード中..."):
        depth_pipe = load_depth_model()

    with st.spinner("深度推定を実行中..."):
        depth_result = depth_pipe(img_pil)

    depth_map = np.array(depth_result["depth"]).astype("float32")

    # --------------------------------------------------
    # 深度マップを 0〜1 に正規化（NumPy 2.0 対応）
    # --------------------------------------------------
    depth_min = float(depth_map.min())
    depth_range = float(np.ptp(depth_map))  # max - min
    if depth_range < 1e-6:  # 万が一すべて同じ値なら
        depth_range = 1.0

    depth_norm = (depth_map - depth_min) / (depth_range + 1e-8)
    col2.image(depth_norm, caption="深度マップ（正規化）", width="stretch")

    # --------------------------------------------------
    # 3D 点群生成（奥行き強調 + 正規化 + 点数制限）
    # --------------------------------------------------
    st.subheader("🌐 3D 点群表示（奥行き + 透視風変換）")

    # 画像座標グリッド
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map.astype(float)

    # 間引き処理
    xs = x[::step, ::step].flatten()
    ys = y[::step, ::step].flatten()
    zs = z[::step, ::step].flatten()

    # RGB 色
    colors = img_rgb[::step, ::step].reshape(-1, 3) / 255.0

    # 点数が多すぎる場合はランダムサンプリングして WebSocket サイズを抑える
    total_points = len(zs)
    if total_points > max_points:
        idx = np.random.choice(total_points, max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
        zs = zs[idx]
        colors = colors[idx]

    # --- 奥行きの指数強調 ---
    zs = zs ** exp_factor

    # --- 奥行きスケール ---
    zs = zs * z_scale

    # --- x, y, z を同じスケールに正規化して「立方体」に収める ---
    xs = xs.astype(float)
    ys = ys.astype(float)

    xs -= xs.mean()
    ys -= ys.mean()
    zs -= zs.mean()

    span_x = np.ptp(xs) + 1e-6
    span_y = np.ptp(ys) + 1e-6
    span_z = np.ptp(zs) + 1e-6

    max_span = max(span_x, span_y, span_z)

    xs = xs / max_span * 1000.0
    ys = ys / max_span * 1000.0
    zs = zs / max_span * 1000.0

    st.caption(f"点群数: {len(zs):,} 点（step={step}, 最大 {max_points:,} 点）")

    # --------------------------------------------------
    # Plotly で 3D 点群を描画
    # --------------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker=dict(
                size=2,
                color=colors,
            ),
        )
    )

    fig.update_layout(
        title="3D 点群ビューア（Depth Anything ベース）",
        scene=dict(
            aspectmode="cube",  # x,y,z を同じスケールに
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        height=750,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, width="stretch")
