"""
文件名: eval.py
所属路径: src/bin

功能概述:
    脚本用于对字节级语言模型 (RWKV-x070) 进行评估，计算负对数似然 (NLL) 及多种频率/熵修正得分，并可输出可视化结果。

主要流程:
    1. 加载评估文件并按 ctx_len+1 字节段切分。
    2. 调用 `RWKV_x070` 模型执行推理，收集概率分布。
    3. 计算原始困惑度、频率修正、上下文熵修正等多种得分。
    4. 可选绘制分数曲线并标注 UTF-8 字符对齐的标签。

关键依赖:
    - src.infering_loop.inferer.RWKV_x070
    - src.datasets.dataset_pt.MyDataset (用于可能的批量评估)

命令行参数 (重要示例):
    --ctx_len (int)            : 上下文窗口长度
    --freq_alpha (float)       : 低频字节抑制系数
    --ctx_window (int)         : 计算滑动熵时的窗口半径

"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import codecs
import csv
import torch.nn.functional as F

from src.datasets.dataset_pt import MyDataset
from src.infering_loop.inferer import RWKV_x070


UTF8_PADDING_CHAR = "↳"
UTF8_REPLACEMENT_CHAR = "\ufffd"


def bytes_to_text(u8_list: list[int]) -> str:
    """将字节序列安全解码为 UTF-8 文本

    功能说明:
        1. 过滤非 0-255 范围值；
        2. 去除前缀 continuation bytes；
        3. 使用增量解码避免尾部截断导致的异常；

    参数:
        u8_list (list[int]): 字节列表，元素需在 0-255 之间。

    返回:
        str: 解码后的文本，遇到非法序列使用 "" 代替。
    """
    start = 0
    temp_u8_list = []
    for item in u8_list:
        if isinstance(item, int) and 0 <= item <= 255:
            temp_u8_list.append(item)

    u8_list = temp_u8_list

    while start < len(u8_list) and (u8_list[start] & 0b1100_0000) == 0b1000_0000:
        start += 1

    byte_seq = bytes(u8_list[start:])

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(byte_seq, final=False)

    text += decoder.decode(b"", final=True)
    return text


def generate_byte_aligned_labels(byte_list: list[int]) -> list[str]:
    """生成与字节序列等长的字符标签列表，用于 Matplotlib X 轴对齐。

    规则:
        • 有效 UTF-8 多字节字符在首字节处标记实际字符，其余字节填充 `UTF8_PADDING_CHAR`；
        • 不合法或截断字节用 `UTF8_REPLACEMENT_CHAR` 填充。
    """
    labels = [""] * len(byte_list)
    i = 0
    while i < len(byte_list):
        byte = byte_list[i]
        char_len = 0

        if not (isinstance(byte, int) and 0 <= byte <= 255):
            labels[i] = UTF8_REPLACEMENT_CHAR
            i += 1
            continue

        if (byte & 0x80) == 0x00:
            char_len = 1
        elif (byte & 0xE0) == 0xC0:
            char_len = 2
        elif (byte & 0xF0) == 0xE0:
            char_len = 3
        elif (byte & 0xF8) == 0xF0:
            char_len = 4
        else:
            labels[i] = UTF8_REPLACEMENT_CHAR
            i += 1
            continue

        if i + char_len > len(byte_list):
            labels[i] = UTF8_REPLACEMENT_CHAR
            i += 1
            continue

        byte_chunk = bytes(byte_list[i : i + char_len])
        try:

            decoded_char = byte_chunk.decode("utf-8")

            labels[i] = decoded_char[0]
            for j in range(1, char_len):
                labels[i + j] = UTF8_PADDING_CHAR
            i += char_len
        except UnicodeDecodeError:
            labels[i] = UTF8_REPLACEMENT_CHAR

            i += 1

    return labels


def main(args):
    """评估入口函数。

    参数:
        args (argparse.Namespace): 命令行解析结果，需包含 ctx_len, freq_alpha, ctx_window 等字段。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    freq_csv_path = os.path.join("assets", "byte_freq.csv")
    freq_arr = np.zeros(256, dtype=np.float64)
    if os.path.isfile(freq_csv_path):
        with open(freq_csv_path, newline="", encoding="utf-8") as f_csv:
            reader = csv.reader(f_csv)
            for row in reader:
                if not row:
                    continue
                try:
                    byte_id = int(row[0])
                    freq_val = float(row[1])
                    if 0 <= byte_id < 256:
                        freq_arr[byte_id] = freq_val
                except Exception:

                    pass

    f_max = float(freq_arr.max()) if freq_arr.max() > 0 else 1.0
    freq_normalized = freq_arr / f_max

    freq_tensor = torch.tensor(freq_normalized, dtype=torch.float32, device=device)

    freq_alpha = args.freq_alpha
    ctx_window = args.ctx_window
    eps_const = 1e-6

    eval_file_path = getattr(args, "eval_file", os.path.join("assets", "ddn.md"))
    with open(eval_file_path, "rb") as f:
        file_bytes = list(f.read())

    if len(file_bytes) < 2:
        print(f"[ERROR] 评估文件 {eval_file_path} 过短(<2 bytes)，无法计算 NLL！")
        return

    print(f"已从 {eval_file_path} 读取 {len(file_bytes)} 字节，用于评估。")

    ctx_len = args.ctx_len
    segments: list[list[int]] = []
    pos = 0
    while pos + 2 <= len(file_bytes) and len(segments) < args.num_samples_to_eval:
        end_pos = min(pos + ctx_len + 1, len(file_bytes))
        seg = file_bytes[pos:end_pos]
        if len(seg) >= 2:
            segments.append(seg)

        pos = end_pos - 1
        break

    args.vocab_size = 256
    model = RWKV_x070(args)
    model.eval()

    all_scores_final: list[list[float]] = []
    all_scores_no_freq: list[list[float]] = []
    all_scores_no_ctx: list[list[float]] = []
    all_scores_raw: list[list[float]] = []
    all_cleaned_tokens_per_sample: list[list[int]] = []

    print(f"开始评估，共 {len(segments)} 段……")
    with torch.no_grad():
        for i, seg in enumerate(segments):
            input_tokens = seg[:-1]
            target_tokens = seg[1:]

            logits, _ = model(input_tokens, None, True)
            probs = torch.softmax(logits.float(), dim=-1)

            targets_tensor = torch.tensor(
                target_tokens, device=probs.device, dtype=torch.long
            )

            prob_of_target = probs[torch.arange(len(target_tokens)), targets_tensor]
            s_i_tensor = -torch.log(prob_of_target + 1e-9)

            entropies_tensor = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

            kernel_size = 2 * ctx_window + 1
            kernel = torch.ones(
                1, 1, kernel_size, device=entropies_tensor.device
            ) / float(kernel_size)
            ent_padded = F.pad(
                entropies_tensor.unsqueeze(0).unsqueeze(0),
                (ctx_window, ctx_window),
                mode="replicate",
            )
            ctx_mean_entropy = F.conv1d(ent_padded, kernel).squeeze()

            w_tensor = torch.pow(freq_tensor[targets_tensor] + eps_const, freq_alpha)

            s_final_tensor = w_tensor * s_i_tensor / (ctx_mean_entropy + eps_const)

            s_no_freq_tensor = s_i_tensor / (ctx_mean_entropy + eps_const)

            s_no_ctx_tensor = w_tensor * s_i_tensor

            s_raw_tensor = s_i_tensor

            all_scores_final.append(s_final_tensor.cpu().tolist())
            all_scores_no_freq.append(s_no_freq_tensor.cpu().tolist())
            all_scores_no_ctx.append(s_no_ctx_tensor.cpu().tolist())
            all_scores_raw.append(s_raw_tensor.cpu().tolist())

            all_cleaned_tokens_per_sample.append(target_tokens)

            print(f"已处理段 {i + 1}/{len(segments)} (长度={len(target_tokens)})")

    threshold_list = getattr(args, "scan_thresholds", [8.0, 16.0, 32.0, 64.0, 128.0])
    _base_colors = [
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "cyan",
        "magenta",
        "gray",
        "olive",
        "pink",
    ]
    threshold_colors = _base_colors[: len(threshold_list)]

    def calc_boundaries(score_seq: list[float], T: float):

        bounds = []
        acc = 0.0
        for idx, val in enumerate(score_seq):
            acc += val
            if acc >= T:
                bounds.append(idx)
                acc = 0.0
        return bounds

    print("Starting to generate charts...")
    for i in range(min(len(all_scores_final), 5)):
        current_sample_nlls_plot = all_scores_final[i]
        corresponding_cleaned_tokens = all_cleaned_tokens_per_sample[i]

        plot_labels_for_nlls = generate_byte_aligned_labels(
            corresponding_cleaned_tokens
        )

        if len(plot_labels_for_nlls) != len(current_sample_nlls_plot):
            print(
                f"ERROR: Length mismatch for sample {i+1}. NLLs: {len(current_sample_nlls_plot)}, Labels: {len(plot_labels_for_nlls)}. Adjusting labels."
            )
            if len(plot_labels_for_nlls) > len(current_sample_nlls_plot):
                plot_labels_for_nlls = plot_labels_for_nlls[
                    : len(current_sample_nlls_plot)
                ]
            else:
                plot_labels_for_nlls.extend(
                    [UTF8_REPLACEMENT_CHAR]
                    * (len(current_sample_nlls_plot) - len(plot_labels_for_nlls))
                )

        text_for_sample_y_display_console = bytes_to_text(corresponding_cleaned_tokens)

        SUBPLOT_TARGET_TOKENS = 256
        MAX_LABELS_PER_SUBPLOT = 256
        ANNOTATION_ROTATION = 0

        variants_plot = [
            ("final", all_scores_final[i], "dodgerblue"),
            ("no_freq", all_scores_no_freq[i], "darkorange"),
            ("no_ctx", all_scores_no_ctx[i], "seagreen"),
        ]

        nlls_to_plot_main = variants_plot[0][1]
        labels_to_plot_main = plot_labels_for_nlls

        bounds_by_threshold = {
            T: calc_boundaries(nlls_to_plot_main, T) for T in threshold_list
        }

        if len(nlls_to_plot_main) <= MAX_LABELS_PER_SUBPLOT:

            plot_width = 17
            current_fig = plt.figure(figsize=(plot_width, 7))
            ax_main = plt.gca()

            for name, seq, color in variants_plot:
                ax_main.plot(seq, linestyle="-", linewidth=1, color=color, label=name)
            ax_main.set_title(
                f"NLL per Token for Sample {i + 1} (Mean NLL: {np.mean(nlls_to_plot_main):.2f})",
                fontsize=14,
            )

            ax_main.set_xticks(range(len(nlls_to_plot_main)))
            ax_main.set_xticklabels(
                labels_to_plot_main,
                rotation=ANNOTATION_ROTATION,
                ha="center",
                va="top",
                fontsize=7,
            )

            plt.subplots_adjust(bottom=0.2)

            print(
                f"DEBUG: Set {len(labels_to_plot_main)} x-axis labels for single plot sample {i+1}"
            )

            ax_main.set_ylabel("NLL (Proxy for Entropy)", fontsize=12)
            ax_main.set_xlabel("Token Index / Character", fontsize=12)
            ax_main.grid(True, linestyle=":", alpha=0.6)

            for idx_T, T in enumerate(threshold_list):
                _color = threshold_colors[idx_T % len(threshold_colors)]
                for k, b_idx in enumerate(bounds_by_threshold[T]):
                    ax_main.axvline(
                        x=b_idx,
                        color=_color,
                        linestyle="--",
                        linewidth=1,
                        alpha=0.8,
                        label=f"T={T}" if k == 0 else None,
                    )
            ax_main.legend()
            current_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        else:
            num_s_plots = int(np.ceil(len(nlls_to_plot_main) / SUBPLOT_TARGET_TOKENS))

            print(
                f"DEBUG: Creating {num_s_plots} subplots for {len(nlls_to_plot_main)} tokens (target: {SUBPLOT_TARGET_TOKENS} per subplot)"
            )

            fig_height_per_subplot = 4.0
            fig_total_height = min(fig_height_per_subplot * num_s_plots, 200)
            fig_width = 25

            current_fig, axs = plt.subplots(
                num_s_plots,
                1,
                figsize=(fig_width, fig_total_height),
                sharey=True,
                squeeze=False,
            )
            axs = axs.flatten()

            variant_chunks = {
                name: np.array_split(np.array(seq), num_s_plots)
                for name, seq, _ in variants_plot
            }
            nlls_chunks = variant_chunks["final"]
            labels_chunks = np.array_split(np.array(labels_to_plot_main), num_s_plots)

            current_fig.suptitle(
                f"NLL per Token for Sample {i + 1} (Mean NLL: {np.mean(nlls_to_plot_main):.2f}) - {len(nlls_to_plot_main)} tokens in {num_s_plots} subplots",
                fontsize=16,
            )

            chunk_start_idx_val = 0
            total_labels_set = 0
            for j, ax_j in enumerate(axs):
                chunk_nlls_data = nlls_chunks[j].tolist()
                chunk_labels_data = labels_chunks[j].tolist()

                global_start_idx = chunk_start_idx_val
                global_end_idx = global_start_idx + len(chunk_nlls_data) - 1
                chunk_start_idx_val += len(chunk_nlls_data)

                for name, seq, color in variants_plot:
                    chunk_seq = variant_chunks[name][j].tolist()
                    ax_j.plot(
                        chunk_seq,
                        linewidth=1,
                        color=color,
                        label=name if j == 0 else None,
                    )
                ax_j.set_title(
                    f"Tokens {global_start_idx}-{global_end_idx} (Mean NLL: {np.mean(chunk_nlls_data):.2f})",
                    fontsize=9,
                )
                ax_j.grid(True, linestyle=":", alpha=0.6)

                if j == num_s_plots // 2 or num_s_plots == 1:
                    ax_j.set_ylabel("NLL", fontsize=9)
                if j == num_s_plots - 1:
                    ax_j.set_xlabel("Token Index / Character", fontsize=9)

                ax_j.set_xticks(range(len(chunk_nlls_data)))
                ax_j.set_xticklabels(
                    chunk_labels_data,
                    rotation=ANNOTATION_ROTATION,
                    ha="center",
                    va="top",
                    fontsize=5,
                )
                total_labels_set += len(chunk_labels_data)

                if j == 0:
                    print(
                        f"DEBUG: First subplot (0): {len(chunk_nlls_data)} tokens, labels: {chunk_labels_data[:5]}..."
                    )
                elif j == num_s_plots - 1:
                    print(
                        f"DEBUG: Last subplot ({j}): {len(chunk_nlls_data)} tokens, labels: {chunk_labels_data[:5]}..."
                    )

                for idx_T, T in enumerate(threshold_list):
                    _color = threshold_colors[idx_T % len(threshold_colors)]
                    local_bounds = [
                        b - global_start_idx
                        for b in bounds_by_threshold[T]
                        if global_start_idx <= b <= global_end_idx
                    ]
                    for k_local, b_local in enumerate(local_bounds):
                        ax_j.axvline(
                            x=b_local,
                            color=_color,
                            linestyle="--",
                            linewidth=1,
                            alpha=0.8,
                            label=f"T={T}" if (j == 0 and k_local == 0) else None,
                        )

            print(
                f"DEBUG: Set {total_labels_set} total x-axis labels across {num_s_plots} subplots for sample {i+1} (target: {SUBPLOT_TARGET_TOKENS} per subplot)"
            )

            if num_s_plots > 0:
                axs[0].legend()
            current_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if current_fig is not None:
            plot_path = os.path.join(args.output_dir, f"nll_sample_{i + 1}.png")
            try:
                current_fig.savefig(plot_path)
                print(f"Chart saved to: {plot_path}")
            except Exception as e:
                print(f"\033[91mFailed to save chart {plot_path}: {e}\033[0m")
            plt.close(current_fig)
        else:
            print(f"No figure was created for sample {i+1}, skipping save.")

    if all_scores_final and any(len(s) > 0 for s in all_scores_final):
        flat_nlls_for_dist = np.concatenate([s for s in all_scores_final if len(s) > 0])
        if flat_nlls_for_dist.size > 0:
            plt.figure(figsize=(12, 7))
            plt.hist(
                flat_nlls_for_dist,
                bins=max(
                    50,
                    (
                        len(set(flat_nlls_for_dist)) // 5
                        if len(set(flat_nlls_for_dist)) > 25
                        else 10
                    ),
                ),
                color="skyblue",
                edgecolor="black",
                alpha=0.75,
            )
            plt.xlabel("NLL (Proxy for Entropy)", fontsize=12)
            plt.ylabel("Frequency (Log Scale)", fontsize=12)
            plt.yscale("log")
            plt.title(
                f"NLL Distribution for All Evaluated Tokens (Total {len(flat_nlls_for_dist)} tokens)",
                fontsize=14,
            )
            mean_nll_total = np.mean(flat_nlls_for_dist)
            median_nll_total = np.median(flat_nlls_for_dist)
            plt.axvline(
                mean_nll_total,
                color="red",
                linestyle="dashed",
                linewidth=1.5,
                label=f"Mean: {mean_nll_total:.2f}",
            )
            plt.axvline(
                median_nll_total,
                color="green",
                linestyle="dashed",
                linewidth=1.5,
                label=f"Median: {median_nll_total:.2f}",
            )
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            dist_plot_path = os.path.join(args.output_dir, "nll_distribution.png")
            try:
                plt.savefig(dist_plot_path)
                print(f"Distribution chart saved to: {dist_plot_path}")
            except Exception as e:
                print(
                    f"\033[91mFailed to save distribution chart {dist_plot_path}: {e}\033[0m"
                )
            plt.close()
        else:
            print(
                "No NLL values collected to plot distribution chart (after filtering empty samples)."
            )
    else:
        print("No NLL values collected to plot distribution chart.")

    print(f"Evaluation finished. Charts saved to directory: {args.output_dir}")

    threshold_list = getattr(args, "scan_thresholds", [8.0, 16.0, 32.0, 64.0, 128.0])
    threshold_colors = ["red", "green", "orange", "purple"]

    def calc_boundaries(score_seq: list[float], T: float):
        bounds = []
        acc = 0.0
        for idx, val in enumerate(score_seq):
            acc += val
            if acc >= T:
                bounds.append(idx)
                acc = 0.0
        return bounds

    if threshold_list:
        for T in threshold_list:
            lens_all = []
            for seq in all_scores_final:
                bounds = calc_boundaries(seq, T)
                if not bounds:
                    lens_all.append(len(seq))
                else:
                    prev = -1
                    for b in bounds:
                        lens_all.append(b - prev)
                        prev = b
                    if prev < len(seq) - 1:
                        lens_all.append(len(seq) - prev - 1)
            if lens_all:
                print(
                    f"[ChunkStats] T={T}: 平均块长={np.mean(lens_all):.1f} token, 中位数={np.median(lens_all):.1f}, 总块数={len(lens_all)}"
                )
            else:
                print(f"[ChunkStats] T={T}: 无数据")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.MODEL_NAME = "assets/weights/rwkv_s"

    args.output_dir = "out/eval_tokenizer"

    args.epoch_steps = 40320
    args.micro_bsz = 1
    args.real_bsz = 1
    args.ctx_len = 4096

    args.n_layer = 6
    args.n_embd = 256
    args.head_size = 64
    args.vocab_size = 256

    args.num_samples_to_eval = 1024

    args.eval_file = os.path.join("assets", "ddn.md")

    args.freq_alpha = 0.25
    args.ctx_window = 32
    main(args)
