# topk 逐样本 chunk_rate 追踪与数据集汇总统计

## 背景

当前 `RerankCompressor.compress()` 中 `topk` 分支没有记录实际的 chunk_rate。由于 topk 强制包含首尾 chunk（`{0, n-1}`），实际保留比例往往大于配置的 `rate`，需要追踪真实值。

其他 selection mode（topp, cluster, cluster-zscore）已有完整的 rate 追踪机制：
- compressor 中每个样本写入 `rate.csv`，`dataset` 字段为 `{数据集名}_{样本ID}`
- eval_longbench.py 评估结束后调用 `compute_rate_averages()` 汇总为 `avg_rate.csv`

## 设计

### 文件 1：`src/compressor.py`

在 topk 分支（line 765-780）末尾追加 chunk_rate 计算和 CSV 写入逻辑。

**CSV 写入模式**：与 cluster 一致，使用覆盖更新（同一 dataset 行只保留一条）。

**表头**：`["dataset", "chunk_rate"]`（与 topp 一致，topk 不需要 mean 列）。

**新增代码**（在 `selected_chunks = [chunks[i] for i in selected_indices]` 之后）：

```python
chunk_rate = len(selected_chunks) / len(chunks)
chunk_rate_str = f"{chunk_rate:.4f}"
csv_path = os.path.join(result_path, "rate.csv")

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "chunk_rate"])

rows = []
dataset_found = False
with open(csv_path, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

for i, row in enumerate(rows):
    if row[0] == dataset:
        rows[i] = [dataset, chunk_rate_str]
        dataset_found = True
        break

if not dataset_found:
    rows.append([dataset, chunk_rate_str])

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
```

### 文件 2：`experiments/evaluation/eval_longbench.py`

Line 1075 的条件判断中加入 `"topk"`，使 `compute_rate_averages()` 在 topk 模式下也被调用：

```python
# 原代码
run_config["reranker_config"].get("selection_mode") in ["cluster", "topp", "cluster-zscore"]

# 改为
run_config["reranker_config"].get("selection_mode") in ["cluster", "topp", "cluster-zscore", "topk"]
```

## 数据流

1. 每个样本调用 `compress()` → 计算 `chunk_rate = len(selected_chunks) / len(chunks)` → 写入 `rate.csv`
2. 所有样本处理完毕后 → `compute_rate_averages()` 读取 `rate.csv` → 按数据集前缀分组计算均值 → 写入 `avg_rate.csv`

## 注意事项

- topk 的 `rate.csv` 表头只有 `["dataset", "chunk_rate"]`，与 cluster 的 `["dataset", "mean0", "mean1", "chunk_rate"]` 不同。如果同一 `result_path` 下先用 cluster 再用 topk 运行，表头会冲突。但实际使用中一个 result_path 只对应一种 selection_mode，不会出现此问题。
- `compute_rate_averages()` 通过 `rate_column="chunk_rate"` 参数读取列，不依赖表头是否包含 mean 列，因此兼容 topk 的简化表头。
