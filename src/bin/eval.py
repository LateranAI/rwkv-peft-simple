import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import codecs

from src.datasets.dataset import MyDataset
from src.model.rwkv_eval_mode import RWKV_x070

# --- Constants for UTF-8 processing in plot annotations ---
UTF8_PADDING_CHAR = '↳'  # U+21B3 Downwards Arrow with Tip Rightwards
UTF8_REPLACEMENT_CHAR = '\uFFFD' # U+FFFD REPLACEMENT CHARACTER

# --- Existing bytes_to_text function (for console output) ---
def bytes_to_text(u8_list: list[int]) -> str:
    start = 0
    temp_u8_list = []
    for item in u8_list:
        if isinstance(item, int) and 0 <= item <= 255:
            temp_u8_list.append(item)
        # else: optionally log or handle non-byte items if they can occur
    u8_list = temp_u8_list

    while start < len(u8_list) and (u8_list[start] & 0b1100_0000) == 0b1000_0000:
        start += 1
    # No need to print here, this is an internal adjustment
    # if start:
    #     print(f"[bytes_to_text] Dropped {start} leading continuation bytes.")
    
    # Ensure u8_list[start:] is used, not modifying u8_list in place if it's an arg
    byte_seq = bytes(u8_list[start:])

    # ② 增量解码，比直接 errors='replace' 更保险
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = decoder.decode(byte_seq, final=False)  # 末尾可能残缺，设 final=False
    # Incremental decoder might leave partials if final=True is not called at the very end.
    # Calling decode with final=True for any remaining bytes in the buffer.
    text += decoder.decode(b'', final=True) 
    return text

# --- New function to generate byte-aligned labels for plotting ---
def generate_byte_aligned_labels(byte_list: list[int]) -> list[str]:
    labels = [""] * len(byte_list)
    i = 0
    while i < len(byte_list):
        byte = byte_list[i]
        char_len = 0

        if not (isinstance(byte, int) and 0 <= byte <= 255): # Should be pre-cleaned
            labels[i] = UTF8_REPLACEMENT_CHAR
            i += 1
            continue

        # Determine expected UTF-8 character length from start byte
        if (byte & 0x80) == 0x00:  # ASCII
            char_len = 1
        elif (byte & 0xE0) == 0xC0:  # 2-byte sequence
            char_len = 2
        elif (byte & 0xF0) == 0xE0:  # 3-byte sequence
            char_len = 3
        elif (byte & 0xF8) == 0xF0:  # 4-byte sequence
            char_len = 4
        else:  # Invalid start byte (e.g., continuation byte 10xxxxxx or > 0xF4)
            labels[i] = UTF8_REPLACEMENT_CHAR
            i += 1
            continue

        # Check if the expected sequence extends beyond the list length
        if i + char_len > len(byte_list):
            labels[i] = UTF8_REPLACEMENT_CHAR # Mark start byte as error due to truncation
            i += 1 # Advance and process next byte individually
            continue

        byte_chunk = bytes(byte_list[i : i + char_len])
        try:
            # Attempt to decode the chunk
            decoded_char = byte_chunk.decode('utf-8')
            # Ensure it's a single character as expected by UTF-8 rules for start bytes
            # (though .decode() on a valid single char sequence gives 1 char)
            # A more robust check might involve ensuring no REPLACEMENT_CHAR is in decoded_char
            # if the original chunk didn't imply one. But for this context, if it decodes, take it.
            
            labels[i] = decoded_char[0] # Take the first (and ideally only) char
            for j in range(1, char_len):
                labels[i+j] = UTF8_PADDING_CHAR
            i += char_len
        except UnicodeDecodeError:
            labels[i] = UTF8_REPLACEMENT_CHAR
            # If decode fails, only this start byte is marked; subsequent bytes in the
            # supposed sequence will be re-evaluated individually in the next iterations.
            i += 1
            
    return labels

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = MyDataset(args)
    data.global_rank = 0
    args.vocab_size = data.vocab_size
    model = RWKV_x070(args)
    eval_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    
    all_nlls_per_sample = []
    all_cleaned_tokens_per_sample = []  # Store cleaned tokens for each sample
    
    print(f"Starting evaluation for {args.num_samples_to_eval} samples...")
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= args.num_samples_to_eval:
                break
            idx, targets_from_batch = batch
            if i == 0:  # Print shape only for the first batch
                print("targets_from_batch shape:", targets_from_batch.shape)

            idx = idx.squeeze().cpu().detach().numpy().tolist()
            logits, _ = model(idx, None, True)
            probs = torch.softmax(logits.float(), dim=-1)
            
            current_sample_nlls = []
            
            # 1. Prepare and clean current_sample_token_ids_y
            raw_token_ids_y = targets_from_batch.cpu().view(-1).tolist()
            cleaned_token_ids_for_sample = []
            for tid_val in raw_token_ids_y:
                if isinstance(tid_val, int) and 0 <= tid_val <= 255:
                    cleaned_token_ids_for_sample.append(tid_val)
                else:
                    cleaned_token_ids_for_sample.append(255)  # Replace with a default byte value
            
            targets_for_nll = targets_from_batch.to(device).squeeze()
            if targets_for_nll.ndim > 1:
                targets_for_nll = targets_for_nll.view(-1)

            # Calculate NLLs - ensure lengths match
            min_len = min(probs.shape[0], len(cleaned_token_ids_for_sample), targets_for_nll.shape[0])
            if probs.shape[0] != len(cleaned_token_ids_for_sample):
                print(f"Warning: Length mismatch for sample {i+1}. Probs: {probs.shape[0]}, Cleaned tokens: {len(cleaned_token_ids_for_sample)}. Using {min_len}.")

            for t in range(min_len):
                original_target_token_id_for_nll = targets_for_nll[t].item()
                prob_of_target = probs[t, original_target_token_id_for_nll]
                nll = -torch.log(prob_of_target + 1e-9).item()
                current_sample_nlls.append(nll)

            # Store both NLLs and cleaned tokens for this sample
            all_nlls_per_sample.append(current_sample_nlls)
            all_cleaned_tokens_per_sample.append(cleaned_token_ids_for_sample[:min_len])  # Match NLL length
            
            if (i + 1) % 1 == 0:
                print(f"Processed sample {i + 1}/{args.num_samples_to_eval}")

    # Now generate plots for each sample OUTSIDE the evaluation loop
    print("Starting to generate charts...")
    for i in range(min(len(all_nlls_per_sample), 5)):
        current_sample_nlls_plot = all_nlls_per_sample[i]
        corresponding_cleaned_tokens = all_cleaned_tokens_per_sample[i]
        
        # 2. Generate byte-aligned labels for plot annotations
        plot_labels_for_nlls = generate_byte_aligned_labels(corresponding_cleaned_tokens)

        # Ensure lengths match before plotting
        if len(plot_labels_for_nlls) != len(current_sample_nlls_plot):
            print(f"ERROR: Length mismatch for sample {i+1}. NLLs: {len(current_sample_nlls_plot)}, Labels: {len(plot_labels_for_nlls)}. Adjusting labels.")
            if len(plot_labels_for_nlls) > len(current_sample_nlls_plot):
                plot_labels_for_nlls = plot_labels_for_nlls[:len(current_sample_nlls_plot)]
            else:
                plot_labels_for_nlls.extend([UTF8_REPLACEMENT_CHAR] * (len(current_sample_nlls_plot) - len(plot_labels_for_nlls)))
        
        # Verify alignment with detailed debug info
        # print(f"DEBUG: Sample {i+1} data verification:")
        # print(f"  - Total NLLs: {len(current_sample_nlls_plot)}")
        # print(f"  - Total labels: {len(plot_labels_for_nlls)}")
        # print(f"  - First 10 NLLs: {current_sample_nlls_plot[:10]}")
        # print(f"  - First 10 labels: {plot_labels_for_nlls[:10]}")
        # print(f"  - Last 10 NLLs: {current_sample_nlls_plot[-10:]}")
        # print(f"  - Last 10 labels: {plot_labels_for_nlls[-10:]}")
        # print(f"  - Cleaned tokens length: {len(corresponding_cleaned_tokens)}")
        # print(f"  - First 10 cleaned tokens (bytes): {corresponding_cleaned_tokens[:10]}")
        # print(f"  - Last 10 cleaned tokens (bytes): {corresponding_cleaned_tokens[-10:]}")
        
        # For console printing (using the same cleaned tokens for consistency)
        text_for_sample_y_display_console = bytes_to_text(corresponding_cleaned_tokens)
        # print(f"Sample {i+1} Decoded Text (for console, first 200 chars): {text_for_sample_y_display_console[:200]}")
        # print(f"Sample {i+1} Decoded Text (for console, last 200 chars): {text_for_sample_y_display_console[-200:]}")

        # --- Plotting Constants ---
        SUBPLOT_TARGET_TOKENS = 256  # Reduced from 200 to 64 - each subplot shows max 64 characters
        MAX_LABELS_PER_SUBPLOT = 256  # New: explicit limit for readable labels
        ANNOTATION_ROTATION = 0  # Changed from 45 to 30 for better readability
        
        # Debug: Print first few labels to verify they're being generated
        # print(f"DEBUG: First 10 plot labels for sample {i+1}: {plot_labels_for_nlls[:10]}")
        # print(f"DEBUG: Label stats - Total: {len(plot_labels_for_nlls)}, Non-empty: {sum(1 for l in plot_labels_for_nlls if l and l.strip())}")
        
        # --- Determine plot strategy based on label readability ---
        nlls_to_plot_main = current_sample_nlls_plot
        labels_to_plot_main = plot_labels_for_nlls

        if len(nlls_to_plot_main) <= MAX_LABELS_PER_SUBPLOT:
            # Strategy 1: Short sequence that can display all labels clearly
            # plot_width = max(20, len(nlls_to_plot_main) * 0.8)  # Give each character more space
            plot_width = 17
            current_fig = plt.figure(figsize=(plot_width, 7))
            ax_main = plt.gca()
            ax_main.plot(nlls_to_plot_main, marker='o', linestyle='-', markersize=3, color='dodgerblue', linewidth=1)
            ax_main.set_title(f"NLL per Token for Sample {i + 1} (Mean NLL: {np.mean(nlls_to_plot_main):.2f})", fontsize=14)
            
            # Set x-axis labels using byte-aligned text
            ax_main.set_xticks(range(len(nlls_to_plot_main)))
            ax_main.set_xticklabels(labels_to_plot_main, rotation=ANNOTATION_ROTATION, ha="center", va="top", fontsize=7)
            
            # Adjust bottom margin for rotated text labels
            plt.subplots_adjust(bottom=0.2)
            
            print(f"DEBUG: Set {len(labels_to_plot_main)} x-axis labels for single plot sample {i+1}")

            ax_main.set_ylabel("NLL (Proxy for Entropy)", fontsize=12)
            ax_main.set_xlabel("Token Index / Character", fontsize=12)
            ax_main.grid(True, linestyle=':', alpha=0.6)
            current_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        else: # Strategy 2: Long sequence, use multiple subplots with limited labels each
            num_s_plots = int(np.ceil(len(nlls_to_plot_main) / SUBPLOT_TARGET_TOKENS))
            # REMOVED: min(MAX_SUBPLOTS, num_s_plots) - now show ALL subplots needed
            print(f"DEBUG: Creating {num_s_plots} subplots for {len(nlls_to_plot_main)} tokens (target: {SUBPLOT_TARGET_TOKENS} per subplot)")
            
            fig_height_per_subplot = 4.0  # Reduced slightly since we might have many subplots
            fig_total_height = min(fig_height_per_subplot * num_s_plots, 200)  # Cap total height at 200 inches
            fig_width = 25  # Increased from 20 to give more horizontal space

            current_fig, axs = plt.subplots(num_s_plots, 1, figsize=(fig_width, fig_total_height), sharey=True, squeeze=False)
            axs = axs.flatten() 
            nlls_chunks = np.array_split(np.array(nlls_to_plot_main), num_s_plots)
            labels_chunks = np.array_split(np.array(labels_to_plot_main), num_s_plots)
            
            current_fig.suptitle(f"NLL per Token for Sample {i + 1} (Mean NLL: {np.mean(nlls_to_plot_main):.2f}) - {len(nlls_to_plot_main)} tokens in {num_s_plots} subplots", fontsize=16)
            
            chunk_start_idx_val = 0
            total_labels_set = 0
            for j, ax_j in enumerate(axs):
                chunk_nlls_data = nlls_chunks[j].tolist()
                chunk_labels_data = labels_chunks[j].tolist()
                
                global_start_idx = chunk_start_idx_val
                global_end_idx = global_start_idx + len(chunk_nlls_data) - 1
                chunk_start_idx_val += len(chunk_nlls_data)

                ax_j.plot(chunk_nlls_data, marker='o', linestyle='-', markersize=2, color='dodgerblue', linewidth=1)  # Smaller markers for many subplots
                ax_j.set_title(f"Tokens {global_start_idx}-{global_end_idx} (Mean NLL: {np.mean(chunk_nlls_data):.2f})", fontsize=9)
                ax_j.grid(True, linestyle=':', alpha=0.6)

                if j == num_s_plots // 2 or num_s_plots == 1:
                    ax_j.set_ylabel("NLL", fontsize=9)
                if j == num_s_plots - 1:
                    ax_j.set_xlabel("Token Index / Character", fontsize=9)
                
                # Set x-axis labels for this subplot chunk - should be readable since max 64 chars
                ax_j.set_xticks(range(len(chunk_nlls_data)))
                ax_j.set_xticklabels(chunk_labels_data, rotation=ANNOTATION_ROTATION, ha="center", va="top", fontsize=5)
                total_labels_set += len(chunk_labels_data)
                
                # Debug info for first and last chunks
                if j == 0:
                    print(f"DEBUG: First subplot (0): {len(chunk_nlls_data)} tokens, labels: {chunk_labels_data[:5]}...")
                elif j == num_s_plots - 1:
                    print(f"DEBUG: Last subplot ({j}): {len(chunk_nlls_data)} tokens, labels: {chunk_labels_data[:5]}...")
                
            print(f"DEBUG: Set {total_labels_set} total x-axis labels across {num_s_plots} subplots for sample {i+1} (target: {SUBPLOT_TARGET_TOKENS} per subplot)")
            current_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # --- Common save and close logic ---
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

    # NLL Distribution Plot
    if all_nlls_per_sample and any(len(s) > 0 for s in all_nlls_per_sample):
        flat_nlls_for_dist = np.concatenate([s for s in all_nlls_per_sample if len(s) > 0])
        if flat_nlls_for_dist.size > 0:
            plt.figure(figsize=(12, 7))
            plt.hist(flat_nlls_for_dist, bins=max(50, len(set(flat_nlls_for_dist)) // 5 if len(set(flat_nlls_for_dist)) > 25 else 10),
                     color='skyblue', edgecolor='black', alpha=0.75)
            plt.xlabel("NLL (Proxy for Entropy)", fontsize=12)
            plt.ylabel("Frequency (Log Scale)", fontsize=12)
            plt.yscale('log')
            plt.title(f"NLL Distribution for All Evaluated Tokens (Total {len(flat_nlls_for_dist)} tokens)", fontsize=14)
            mean_nll_total = np.mean(flat_nlls_for_dist)
            median_nll_total = np.median(flat_nlls_for_dist)
            plt.axvline(mean_nll_total, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_nll_total:.2f}')
            plt.axvline(median_nll_total, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_nll_total:.2f}')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            dist_plot_path = os.path.join(args.output_dir, "nll_distribution.png")
            try:
                plt.savefig(dist_plot_path)
                print(f"Distribution chart saved to: {dist_plot_path}")
            except Exception as e:
                print(f"\033[91mFailed to save distribution chart {dist_plot_path}: {e}\033[0m")
            plt.close()
        else:
            print("No NLL values collected to plot distribution chart (after filtering empty samples).")
    else:
        print("No NLL values collected to plot distribution chart.")

    print(f"Evaluation finished. Charts saved to directory: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.MODEL_NAME = "/public/home/ssjxzkz/Projects/block-blm/out/L6-D256-x070/rwkv-211"
    args.data_file = "/public/home/ssjxzkz/Datasets/lm/OptimalScale_ClimbLab/mmap/block_blm_data_device_0"
    args.output_dir = "/public/home/ssjxzkz/Projects/block-blm/out/L6-D256-x070/"

    args.epoch_steps = 40320
    args.micro_bsz = 1
    args.real_bsz = 1
    args.ctx_len = 4096
    args.magic_prime = 1219355507
    args.train_stage = 3

    args.n_layer = 6
    args.n_embd = 256
    args.head_size = 64
    args.vocab_size = 256

    args.num_samples_to_eval = 1024
    main(args)