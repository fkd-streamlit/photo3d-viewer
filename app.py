import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import plotly.graph_objects as go
import tempfile
import os
import uuid

# =====================================================
# Streamlit 基本設定
# =====================================================
st.set_page_config(page_title="Photo3D Viewer", layout="wide")
st.title("📸 Photo3D Viewer – 写真が立体に変わる瞬間")

st.write("""
### 1枚の写真から **AI が奥行きを推定して 3D 点群として可視化** します。
""")

st.markdown("""
## 🌐 Photo3D Viewer へようこそ

このアプリは、1枚の写真からAIを使って **奥行き(Depth)** を推定し、  
3D点群として可視化 します。

---

## 🔧 推奨画像
- 解像度：**横 2000px 以下**  
- 明暗差がある画像は奥行き推定が安定

---

## 📌 使い方（3ステップ）
1. 左側のサイドバーから画像をアップロード  
2. 深度推定が自動実行 → 深度マップが表示  
3. スライダーで奥行き/密度を調整して 3Dビューを楽しむ
""")

# =====================================================
# 📌 画像サイズ(容量)を厳しく制限
# =====================================================
MAX_FILE_SIZE = 2 * 1024 * 1024      # 2MB
MAX_PIXELS = 2000 * 2000             # 400万ピクセル以下

def validate_image(upload):
    if upload.size > MAX_FILE_SIZE:
        return "⚠️ ファイルサイズが大きすぎます（上限2MB）"

    img = Image.open(upload)
    w, h = img.size
    if w * h > MAX_PIXELS:
        return f"⚠️ 画像が大きすぎます（上限 2000×2000px）。現在: {w}×{h}px"

    return None


# =====================================================
# 📌 AIモデル（Depth Anything）をキャッシュして読み込み
# =====================================================
@st.cache_resource
def load_depth_model():
    return pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf")

depth_model = load_depth_model()


# =====================================================
# 📌 サイドバー：画像アップロード
# =====================================================
st.sidebar.header("🖼 画像アップロード")
uploaded = st.sidebar.file_uploader("対応形式：JPG / JPEG / PNG", type=["jpg", "jpeg", "png"])

# スライダーは画像選択後に表示
if uploaded:
    # 安全チェック
    err = validate_image(uploaded)
    if err:
        st.error(err)
        st.stop()

    # 読み込み
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="アップロード画像", use_container_width=True)

    # ===== 深度推定 =====
    with st.spinner("AI が奥行きを推定中..."):
        depth_output = depth_model(img)
        depth = np.array(depth_output["predicted_depth"])

    st.success("深度マップが生成されました！")
    st.image(depth, caption="深度マップ", use_container_width=True, clamp=True)

    # ===== 3Dビュー設定 =====
    st.subheader("🎛 奥行き・点群の調整")
    exp_factor = st.slider("奥行き強調倍率", 0.5, 5.0, 1.5, 0.1)
    z_scale = st.slider("奥行きのスケール", 0.5, 3.0, 1.0, 0.1)
    max_points = st.slider("点群密度（ポイント数）", 5000, 30000, 15000, 1000)

    # 点群生成
    h, w = depth.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth * z_scale * exp_factor

    # flatten
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()

    # ランダムサンプリング
    idx = np.random.choice(len(Xf), size=max_points, replace=False)
    Xs, Ys, Zs = Xf[idx], Yf[idx], Zf[idx]

    # ===== Plotly 3D =====
    fig = go.Figure(data=[
        go.Scatter3d(
            x=Xs, y=Ys, z=Zs,
            mode="markers",
            marker=dict(
                size=2,
                color=Zs,
                colorscale="Viridis",
            )
        )
    ])

    fig.update_layout(
        width=900,
        height=700,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        )
    )

    st.plotly_chart(fig, use_container_width=True)
