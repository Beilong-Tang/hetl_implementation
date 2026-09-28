"""为论文的三个迁移任务构造可重复的 NSL-KDD 数据。

完整的数据流：

    KDDTrain+.txt（每行 43 列）
      = 41 个网络特征 + attack + difficulty
                    |
                    | 1. 清洗 attack 并映射到攻击大类
                    | 2. difficulty 不进入模型特征
                    | 3. 构造 source domain 和 target domain
                    | 4. 文本类别特征执行 one-hot 编码
                    | 5. 删除两个域中都没有变化的常量列
                    | 6. 两个域分别执行 Min-Max 归一化
                    v
    NPZ 文件：X_source、y_source、X_target、y_target、划分索引、特征名

原始数据始终保持只读；处理结果只写入 data/processed 和 data/splits。

运行方式：
    uv run python src/preprocessing.py
    uv run python src/preprocessing.py --all-seeds

默认只处理随机种子 42，便于先检查流程。加 ``--all-seeds`` 后会处理
configs/experiment.json 中的全部随机种子。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN = ROOT / "data" / "raw" / "KDDTrain+.txt"
CONFIG_PATH = ROOT / "configs" / "experiment.json"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
# TXT 文件没有表头，因此必须严格按照 NSL-KDD 的官方顺序指定列名。
# 前 41 列是网络特征；attack 是具体攻击名称；difficulty 是数据集制作阶段
# 根据历史分类器表现得到的样本难度信息，而不是实际网络中可直接测量的特征。
# 所以 difficulty 只在读取时保留用于检查，绝不会进入模型输入矩阵。
COLUMNS = FEATURE_COLUMNS + ["attack", "difficulty"]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]

ATTACK_GROUPS = {
    # DoS
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",
    "worm": "dos",
    # Probe
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "satan": "probe",
    "mscan": "probe",
    "saint": "probe",
    # R2L
    "ftp_write": "r2l",
    "guess_passwd": "r2l",
    "imap": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "httptunnel": "r2l",
    "named": "r2l",
    "sendmail": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    # U2R
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "perl": "u2r",
    "rootkit": "u2r",
    "ps": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
}

EXPECTED_TRAIN_COUNTS = {
    "normal": 67_343,
    "dos": 45_927,
    "probe": 11_656,
    "r2l": 995,
    "u2r": 52,
}


def load_config() -> dict:
    """读取项目实验配置。"""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_training_data() -> pd.DataFrame:
    """读取 KDDTrain+，清洗标签，并核验论文 Table I 的类别数量。"""
    if not RAW_TRAIN.is_file():
        raise FileNotFoundError(f"找不到原始训练集：{RAW_TRAIN}")

    frame = pd.read_csv(RAW_TRAIN, names=COLUMNS, header=None)
    if frame.shape != (125_973, 43):
        raise ValueError(f"训练集形状异常：{frame.shape}，预期 (125973, 43)")

    # raw_index 只用于追踪记录来自原文件的哪一行，不属于模型特征。
    # 有了它，就能检查源域/目标域是否重复使用了同一条正常记录。
    frame.insert(0, "raw_index", np.arange(len(frame), dtype=np.int64))
    frame["attack"] = (
        frame["attack"].astype(str).str.strip().str.rstrip(".").str.lower()
    )
    frame["attack_group"] = frame["attack"].map(ATTACK_GROUPS)
    frame.loc[frame["attack"] == "normal", "attack_group"] = "normal"

    unknown = sorted(frame.loc[frame["attack_group"].isna(), "attack"].unique())
    if unknown:
        raise ValueError(f"存在未映射的攻击标签：{unknown}")

    actual_counts = frame["attack_group"].value_counts().to_dict()
    if actual_counts != EXPECTED_TRAIN_COUNTS:
        raise ValueError(
            "类别数量与论文 Table I 不一致。\n"
            f"实际：{actual_counts}\n预期：{EXPECTED_TRAIN_COUNTS}"
        )
    return frame


def sample_domain_rows(
    frame: pd.DataFrame,
    source_attack: str,
    target_attack: str,
    samples_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造等规模、类别平衡且正常样本不重叠的源域和目标域。"""
    rng = np.random.default_rng(seed)
    normal_indices = frame.index[frame["attack_group"] == "normal"].to_numpy()
    source_attack_indices = frame.index[
        frame["attack_group"] == source_attack
    ].to_numpy()
    target_attack_indices = frame.index[
        frame["attack_group"] == target_attack
    ].to_numpy()

    if len(normal_indices) < 2 * samples_per_class:
        raise ValueError("正常样本不足，无法保证两个域不重叠")
    if len(source_attack_indices) < samples_per_class:
        raise ValueError(f"{source_attack} 样本不足 {samples_per_class} 条")
    if len(target_attack_indices) < samples_per_class:
        raise ValueError(f"{target_attack} 样本不足 {samples_per_class} 条")

    selected_normal = rng.choice(normal_indices, 2 * samples_per_class, replace=False)
    source_indices = np.concatenate(
        [
            selected_normal[:samples_per_class],
            rng.choice(source_attack_indices, samples_per_class, replace=False),
        ]
    )
    target_indices = np.concatenate(
        [
            selected_normal[samples_per_class:],
            rng.choice(target_attack_indices, samples_per_class, replace=False),
        ]
    )
    rng.shuffle(source_indices)
    rng.shuffle(target_indices)

    source = frame.loc[source_indices].copy().reset_index(drop=True)
    target = frame.loc[target_indices].copy().reset_index(drop=True)
    source["binary_label"] = (source["attack_group"] != "normal").astype(np.int8)
    target["binary_label"] = (target["attack_group"] != "normal").astype(np.int8)

    source_normal = set(source.loc[source["binary_label"] == 0, "raw_index"])
    target_normal = set(target.loc[target["binary_label"] == 0, "raw_index"])
    if source_normal & target_normal:
        raise AssertionError("源域和目标域出现重复的正常记录")
    return source, target


def encode_and_scale(
    source: pd.DataFrame, target: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """共同建立 one-hot 字典，然后对两个域分别执行 Min-Max 归一化。

    One-hot 原理：
        protocol_type、service、flag 是名称，不存在数值大小关系。例如不能把
        tcp、udp、icmp 简单写成 0、1、2，否则模型会误以为 2 大于 1。
        one-hot 会生成 protocol_type_tcp、protocol_type_udp、
        protocol_type_icmp 等列；一条记录属于哪个类别，对应列就是 1，
        其他列为 0。

    共同拟合编码器的原因：
        HeTL 是传导式域适应，允许查看无标签目标域的特征。联合建立类别字典
        可以保证 X_source 和 X_target 的列数、顺序和含义完全一致。此处没有
        读取目标域标签，因此不构成标签泄漏。

    Min-Max 原理：
        对每一列执行 x_scaled = (x - min) / (max - min)，通常将数值映射到
        0~1。NSL-KDD 的 rate、bytes、duration、count 量纲相差很大；归一化
        可防止 bytes 等大数值特征支配距离、重构误差和梯度。

    返回：
        source_array: (源域样本数, 编码后特征数)
        target_array: (目标域样本数, 编码后特征数)
        feature_names: 每一列对应的可读特征名
    """
    # sparse_output=False 返回普通二维数组，便于后续 HeTL 直接做矩阵运算。
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    # 这里只读取三个类别特征，不读取 binary_label。
    encoder.fit(pd.concat([source[CATEGORICAL_COLUMNS], target[CATEGORICAL_COLUMNS]]))

    source_numeric = source[NUMERIC_COLUMNS].astype(np.float64).to_numpy()
    target_numeric = target[NUMERIC_COLUMNS].astype(np.float64).to_numpy()
    source_categorical = encoder.transform(source[CATEGORICAL_COLUMNS])
    target_categorical = encoder.transform(target[CATEGORICAL_COLUMNS])

    # 最终列顺序为：38 个数值/二值原始特征 + one-hot 展开的类别列。
    # attack、attack_group、binary_label、difficulty、raw_index 都不会进入矩阵。
    source_array = np.column_stack([source_numeric, source_categorical])
    target_array = np.column_stack([target_numeric, target_categorical])
    feature_names = np.asarray(
        NUMERIC_COLUMNS + encoder.get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
    )

    # 仅删除两个域中都为常量的列，保证两边仍有相同特征空间。
    keep_mask = (np.ptp(source_array, axis=0) > 0) | (np.ptp(target_array, axis=0) > 0)
    source_array = source_array[:, keep_mask]
    target_array = target_array[:, keep_mask]
    feature_names = feature_names[keep_mask]

    # 对应论文算法中的 "Normalize T, S"。两个 scaler 只读取特征，不读取标签。
    # 这里选择两个域分别归一化；这是论文未说明细节时采用的可复现实现选择。
    source_array = MinMaxScaler().fit_transform(source_array)
    target_array = MinMaxScaler().fit_transform(target_array)
    return source_array, target_array, feature_names


def save_task(
    frame: pd.DataFrame,
    source_attack: str,
    target_attack: str,
    samples_per_class: int,
    target_validation_size: int,
    seed: int,
) -> dict:
    """处理并保存一个 source -> target 任务。"""
    task_name = f"{source_attack}_to_{target_attack}_seed_{seed}"
    source, target = sample_domain_rows(
        frame, source_attack, target_attack, samples_per_class, seed
    )
    source_array, target_array, feature_names = encode_and_scale(source, target)

    source_labels = source["binary_label"].to_numpy(dtype=np.int8)
    target_labels = target["binary_label"].to_numpy(dtype=np.int8)
    all_target_indices = np.arange(len(target_labels))
    validation_indices, test_indices = train_test_split(
        all_target_indices,
        train_size=target_validation_size,
        stratify=target_labels,
        random_state=seed,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED_DIR / f"{task_name}.npz",
        X_source=source_array,
        y_source=source_labels,
        X_target=target_array,
        y_target=target_labels,
        target_validation_indices=validation_indices,
        target_test_indices=test_indices,
        feature_names=feature_names,
    )

    source_split = source[["raw_index", "attack", "attack_group", "binary_label"]].copy()
    source_split.insert(0, "domain", "source")
    source_split["target_partition"] = "not_applicable"

    target_split = target[["raw_index", "attack", "attack_group", "binary_label"]].copy()
    target_split.insert(0, "domain", "target")
    target_split["target_partition"] = "test"
    target_split.loc[validation_indices, "target_partition"] = "validation"
    pd.concat([source_split, target_split], ignore_index=True).to_csv(
        SPLITS_DIR / f"{task_name}.csv", index=False
    )

    return {
        "task": f"{source_attack}->{target_attack}",
        "seed": seed,
        "source_rows": len(source),
        "target_rows": len(target),
        "features_after_encoding": len(feature_names),
        "target_validation_rows": len(validation_indices),
        "target_test_rows": len(test_indices),
        "processed_file": f"data/processed/{task_name}.npz",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="处理配置文件中的全部随机种子；默认只处理第一个种子",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    frame = load_training_data()

    summary = (
        frame.groupby("attack_group", sort=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PROCESSED_DIR / "training_class_summary.csv", index=False)

    seeds = config["random_seeds"] if args.all_seeds else config["random_seeds"][:1]
    results = []
    for seed in seeds:
        for task in config["tasks"]:
            result = save_task(
                frame=frame,
                source_attack=task["source"],
                target_attack=task["target"],
                samples_per_class=config["samples_per_class"],
                target_validation_size=config["target_validation_size"],
                seed=seed,
            )
            results.append(result)
            print(
                f"完成 {result['task']} (seed={seed}): "
                f"source={result['source_rows']}, target={result['target_rows']}, "
                f"features={result['features_after_encoding']}"
            )

    pd.DataFrame(results).to_csv(PROCESSED_DIR / "preprocessing_manifest.csv", index=False)
    print(f"\n处理完成。清单：{PROCESSED_DIR / 'preprocessing_manifest.csv'}")


if __name__ == "__main__":
    main()
