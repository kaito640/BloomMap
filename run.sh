python text_pipe_gpu.py

cp data/48ae6f249b_themes.json proc_themes.json
cp data/48ae6f249b_volumes.json cluster_volumes.json
python3 -m http.server 8000
