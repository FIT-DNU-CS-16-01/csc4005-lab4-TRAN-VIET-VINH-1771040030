# Lab 4: UrbanSound8K — CRNN for Environmental Sound Classification

> **Môn học:** Học sâu (CSC4005)  
> **Họ tên:** Trần Việt Vinh  
> **MSSV:** 1771040030

---

## 📁 Cấu trúc repo

```
csc4005-lab4-TRAN-VIET-VINH-1771040030/
├── README.md
├── REPORT_LAB4.md
├── requirements.txt
├── configs/
│   ├── baseline_logmel_crnn.json
│   ├── fast_debug.json
│   └── extension_bilstm_crnn.json
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── data/
│   └── UrbanSound8K/               ← Dữ liệu (không push lên GitHub)
│       ├── audio/
│       └── metadata/
├── outputs/
│   └── 1771040030_crnn_gru_baseline/
│       ├── best_model.pt
│       ├── history.csv
│       ├── curves.png
│       ├── confusion_matrix.png
│       └── metrics.json
└── wandb/
```

---

## 🛠️ Cài đặt môi trường

### 1. Tạo môi trường Conda

```bash
conda create -n csc4005-dl python=3.10 -y
conda activate csc4005-dl
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Đăng nhập W&B

```bash
wandb login
# Dán API key từ https://wandb.ai/authorize
```

---

## 📂 Chuẩn bị dữ liệu

1. Tải dataset **UrbanSound8K** theo đường dẫn trong đề bài.
2. Giải nén và đặt vào thư mục `data/` theo cấu trúc sau:

```
data/
└── UrbanSound8K/
    ├── audio/
    │   ├── fold1/
    │   ├── fold2/
    │   └── ...fold10/
    └── metadata/
        └── UrbanSound8K.csv
```

> ⚠️ **Lưu ý:** Không push thư mục `data/` lên GitHub (đã có trong `.gitignore`). Dataset ~500MB.

---

## 🚀 Chạy huấn luyện

### 🔹 Debug — kiểm tra pipeline nhanh

```bash
python -m src.train \
  --config configs/fast_debug.json \
  --data_dir ./data/UrbanSound8K \
  --use_wandb
```

### 🔹 Baseline chính — Log-mel + CRNN-GRU

```bash
python -m src.train \
  --config configs/baseline_logmel_crnn.json \
  --data_dir ./data/UrbanSound8K \
  --use_wandb \
  --run_name "1771040030_crnn_gru_baseline"
```

> ⏱️ Baseline mất khoảng **~1.3 giờ** để chạy 25 epochs trên CPU.

### 🔹 (Mở rộng) BiLSTM-CRNN

```bash
python -m src.train \
  --config configs/extension_bilstm_crnn.json \
  --data_dir ./data/UrbanSound8K \
  --use_wandb \
  --run_name "1771040030_bilstm_crnn"
```

> 💡 Nếu gặp lỗi W&B timeout, chạy offline bằng lệnh:
> ```bash
> # PowerShell
> $env:WANDB_MODE = "offline"
> # Bash/Linux
> export WANDB_MODE=offline
> ```

---

## 📊 Kết quả đạt được

### Baseline — Log-mel + CRNN-GRU

| Chỉ số | Giá trị |
|--------|---------|
| Best validation accuracy | **74.51%** |
| Test accuracy | **75.63%** |
| Thời gian trung bình / epoch | 189 giây (~3 phút) |
| Tổng thời gian train | ~1.3 giờ |
| Số lượng tham số | 71,338 |

### So sánh với Lab 3 (1D-CNN)

| Tiêu chí | Lab 3 (1D-CNN) | Lab 4 (CRNN) | Thay đổi |
|----------|----------------|--------------|----------|
| Test accuracy | 52.47% | **75.63%** | ✅ +23.16% |
| Validation accuracy | 59.61% | **74.51%** | ✅ +14.90% |
| Thời gian / epoch | 5 giây | 189 giây | ⚠️ Chậm hơn 38× |
| Số tham số | 137,930 | **71,338** | ✅ Nhẹ hơn |
| Overfitting | Có rõ rệt | Ít hơn | ✅ Cải thiện |

---

## 📈 Outputs sau khi huấn luyện

Kết quả được tự động lưu vào thư mục `outputs/<run_name>/`:

```
outputs/1771040030_crnn_gru_baseline/
├── best_model.pt        # Model checkpoint tốt nhất (theo val accuracy)
├── history.csv          # Loss & accuracy theo từng epoch
├── curves.png           # Learning curves (train/val)
├── confusion_matrix.png # Confusion matrix trên tập test
└── metrics.json         # Tổng hợp metrics cuối cùng
```

---

## ✅ Checklist nộp bài

| STT | Yêu cầu | Trạng thái |
|-----|---------|------------|
| 1 | Repo code chạy được | ✅ |
| 2 | README hướng dẫn chạy | ✅ |
| 3 | Link / ảnh W&B dashboard | ✅ |
| 4 | Kết quả CRNN-GRU | ✅ |
| 5 | `curves.png` | ✅ |
| 6 | `confusion_matrix.png` | ✅ |
| 7 | `metrics.json` | ✅ |
| 8 | Bảng so sánh Lab 3 vs Lab 4 | ✅ |
| 9 | Phân tích lớp dễ bị nhầm | ✅ |

---

## 🔗 W&B Dashboard

| | Link |
|-|------|
| **Project** | [csc4005-lab4-urbansound8k-crnn](https://wandb.ai/vinhtran2785-bt/csc4005-lab4-urbansound8k-crnn) |
| **Run baseline** | [1771040030_crnn_gru_baseline](https://wandb.ai/vinhtran2785-bt/csc4005-lab4-urbansound8k-crnn/runs/t1dt15ez) |

> ⚠️ Cần đăng nhập tài khoản `vinhtran2785` để xem dashboard.

---

## 📝 Ghi chú

- Xem phân tích chi tiết, confusion matrix và trả lời câu hỏi tự kiểm tra tại [`REPORT_LAB4.md`](./REPORT_LAB4.md).
- Để thêm run mới, đặt `--run_name` khác với tên đã có để tránh ghi đè checkpoint.
- Nếu gặp lỗi `CUDA out of memory`, giảm `batch_size` trong file config tương ứng.