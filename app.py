import math
import random

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# 1. 固定随机数，方便重复实验
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# 2. 生成带规律的模拟时序
# ----------------------------
def generate_series(length: int, seed: int):
    rng = np.random.default_rng(seed)

    returns = np.zeros(length, dtype=np.float32)

    for i in range(1, length):
        cycle = 0.002 * math.sin(i / 12)
        momentum = 0.65 * returns[i - 1]
        noise = rng.normal(0, 0.004)

        returns[i] = momentum + cycle + noise

    prices = 100 * np.exp(np.cumsum(returns))

    return returns, prices


# ----------------------------
# 3. 制作滑动窗口训练样本
# ----------------------------
def make_dataset(returns, prices, window_size):
    features = []
    labels = []
    metadata = []

    for end in range(window_size, len(returns) - 1):
        window = returns[end - window_size:end]

        current_price = prices[end]
        next_price = prices[end + 1]

        # 下一个点比当前点高，标签为1；否则为0
        label = 1 if next_price > current_price else 0

        features.append(window)
        labels.append(label)

        metadata.append(
            {
                "时间点": end,
                "当前值": current_price,
                "下一时刻真实值": next_price,
            }
        )

    x = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # 输入形状变成：[样本数, 时间长度, 特征数]
    x = x[:, :, np.newaxis]

    return x, y, metadata


# ----------------------------
# 4. 定义小型 Transformer
# ----------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        window_size,
        d_model=32,
        nhead=4,
        num_layers=2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)

        self.position_embedding = nn.Parameter(
            torch.randn(1, window_size, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding

        encoded = self.transformer(x)

        # 取最后一个时间位置进行分类
        last_token = encoded[:, -1, :]

        return self.classifier(last_token)


# ----------------------------
# 5. 页面配置
# ----------------------------
st.set_page_config(
    page_title="时序 Transformer",
    page_icon="📈",
    layout="wide",
)

st.title("时序模拟数据 Transformer 预测")
st.write("使用过去一段时间的数据，预测下一个时间点会上涨还是下跌。")

with st.sidebar:
    st.header("运行参数")

    series_length = st.slider(
        "时序长度",
        min_value=300,
        max_value=1200,
        value=600,
        step=100,
    )

    window_size = st.slider(
        "输入窗口长度",
        min_value=12,
        max_value=60,
        value=24,
        step=4,
    )

    epochs = st.slider(
        "训练轮数",
        min_value=5,
        max_value=50,
        value=15,
    )

    seed = st.number_input(
        "随机种子",
        min_value=0,
        max_value=9999,
        value=42,
    )

    use_mps = st.checkbox(
        "使用 Apple GPU",
        value=torch.backends.mps.is_available(),
        disabled=not torch.backends.mps.is_available(),
    )


# ----------------------------
# 6. 生成并展示数据
# ----------------------------
set_seed(int(seed))

returns, prices = generate_series(
    length=series_length,
    seed=int(seed),
)

series_df = pd.DataFrame(
    {
        "时间点": np.arange(series_length),
        "模拟数值": prices,
    }
).set_index("时间点")

st.subheader("模拟时序图像")
st.line_chart(series_df, height=320)

x, y, metadata = make_dataset(
    returns=returns,
    prices=prices,
    window_size=window_size,
)

split_index = int(len(x) * 0.8)

x_train = x[:split_index]
y_train = y[:split_index]

x_test = x[split_index:]
y_test = y[split_index:]

test_metadata = metadata[split_index:]

col1, col2, col3 = st.columns(3)

col1.metric("训练样本", len(x_train))
col2.metric("测试样本", len(x_test))
col3.metric("输入窗口", window_size)


# ----------------------------
# 7. 点击按钮开始训练
# ----------------------------
if st.button("开始训练 Transformer", type="primary"):
    device_name = "mps" if use_mps else "cpu"
    device = torch.device(device_name)

    st.info(f"当前计算设备：{device_name.upper()}")

    train_dataset = TensorDataset(
        torch.tensor(x_train),
        torch.tensor(y_train),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    model = TimeSeriesTransformer(
        window_size=window_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    loss_function = nn.CrossEntropyLoss()

    progress_bar = st.progress(0)
    status_area = st.empty()

    chart_column, log_column = st.columns([2, 1])

    loss_chart_area = chart_column.empty()
    log_area = log_column.empty()

    history = []
    logs = []

    # ----------------------------
    # 8. 实时训练
    # ----------------------------
    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        correct = 0
        sample_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = loss_function(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)

            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            sample_total += len(batch_y)

        average_loss = total_loss / sample_total
        training_accuracy = correct / sample_total

        history.append(
            {
                "轮次": epoch,
                "训练损失": average_loss,
                "训练准确率": training_accuracy,
            }
        )

        percent = int(epoch / epochs * 100)
        progress_bar.progress(percent)

        status_area.write(
            f"第 {epoch}/{epochs} 轮｜"
            f"损失：{average_loss:.4f}｜"
            f"训练准确率：{training_accuracy:.2%}"
        )

        history_df = pd.DataFrame(history).set_index("轮次")

        loss_chart_area.line_chart(
            history_df[["训练损失"]],
            height=280,
        )

        logs.append(
            f"Epoch {epoch:02d} | "
            f"Loss {average_loss:.4f} | "
            f"Accuracy {training_accuracy:.2%}"
        )

        log_area.code("\n".join(logs[-10:]))

    st.success("模型训练完成，正在运行测试集预测……")

    # ----------------------------
    # 9. 测试集预测
    # ----------------------------
    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(x_test).to(device)
        test_logits = model(test_tensor)

        probabilities = torch.softmax(test_logits, dim=1)
        predictions = test_logits.argmax(dim=1).cpu().numpy()
        confidence = probabilities.max(dim=1).values.cpu().numpy()

    result_rows = []

    for index, prediction in enumerate(predictions):
        actual = int(y_test[index])
        is_correct = prediction == actual

        row = test_metadata[index]

        result_rows.append(
            {
                "时间点": row["时间点"],
                "当前值": round(row["当前值"], 3),
                "下一时刻真实值": round(
                    row["下一时刻真实值"],
                    3,
                ),
                "实际方向": "上涨" if actual == 1 else "下跌",
                "预测方向": "上涨" if prediction == 1 else "下跌",
                "置信度": f"{confidence[index]:.2%}",
                "预测结果": "✅ 正确" if is_correct else "❌ 错误",
            }
        )

    result_df = pd.DataFrame(result_rows)
    test_accuracy = (predictions == y_test).mean()

    st.subheader("测试结果")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("测试准确率", f"{test_accuracy:.2%}")
    metric2.metric(
        "预测正确",
        int((predictions == y_test).sum()),
    )
    metric3.metric(
        "预测错误",
        int((predictions != y_test).sum()),
    )

    # 用颜色展示预测是否正确
    styled_result = result_df.style.apply(
        lambda row: [
            (
                "background-color: #d1fae5; color: #065f46"
                if row["预测结果"] == "✅ 正确"
                else "background-color: #fee2e2; color: #991b1b"
            )
            if column == "预测结果"
            else ""
            for column in result_df.columns
        ],
        axis=1,
    )

    st.dataframe(
        styled_result,
        use_container_width=True,
        height=500,
    )

    st.subheader("预测正确与错误分布")

    summary_df = pd.DataFrame(
        {
            "数量": [
                int((predictions == y_test).sum()),
                int((predictions != y_test).sum()),
            ]
        },
        index=["预测正确", "预测错误"],
    )

    st.bar_chart(summary_df)