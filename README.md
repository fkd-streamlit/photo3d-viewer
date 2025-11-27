📸 Photo3D Viewer – 写真が立体に変わる瞬間

AI を使って、1枚の写真から 奥行きを推定し、3D 点群として可視化するアプリ です。
Web ブラウザだけで動作し、インストール不要で利用できます。

👉 アプリはこちら：
https://photo3d-viewer.streamlit.app/

🚀 機能
Depth Anything（AIモデル）を使った 深度推定

画像から生成された 3D 点群のインタラクティブ可視化
点群密度・奥行き強調・スケールの調整
WebSocket サイズ制限に応じた点数管理
ブラウザでリアルタイムに回転・拡大・縮小

📥 使い方（3ステップ）
1. サイドバーから画像をアップロード
　　対応形式：JPG / JPEG / PNG
2. 深度推定が自動で実行され、深度マップが表示されます
3. スライダーを調整しながら 3D の立体感を出します

点群密度（数字を小さくすると細かくなる）

奥行き強調（指数）

奥行きスケール
など

📷 推奨画像について
横幅 2000px 以下
人物・建造物など輪郭がはっきりしている画像
明るさにメリハリがある画像はより立体的に再構築されます

🧠 使用している技術
Streamlit
Plotly
OpenCV
Depth Anything (Hugging Face Transformers)
NumPy
Python 3.11

🔧 ローカルで実行する方法
git clone https://github.com/fkd-streamlit/photo3d-viewer.git
cd photo3d-viewer
pip install -r requirements.txt
streamlit run app.py

📄 ライセンス
MIT License（自由に使ってOK）

✨ 作者
福田雅彦（fkd-streamlit）
