# data/

此資料夾用來放**本地生成**的資料集（例如 `*.npz`）。

為了讓 GitHub repo 保持乾淨、避免放入 GB 等級的大檔，本專案預設不追蹤 `data/*.npz`（見 `.gitignore`）。

- 產生資料：`python traindata_collect2p.py`
- 監督式訓練：`python train_nn.py --data data/poker_data_with_baseline_final.npz --out models/policy_nn.pt`


