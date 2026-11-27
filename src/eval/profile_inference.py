from pathlib import Path
import argparse
import time

import pandas as pd
import torch

from src.train.utils import build_model, load_checkpoint


CHECKPOINTS = [
    ("ft_b100_s42", 12, "finetune", "validation-loss"),
    ("ft_b100_s42", 30, "finetune", "Rule W"),
    ("lp_b100_s42", 26, "linear_probe", "linear probe"),
]


def profile_model(model, device, warmup=10, timed=100):
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        times = []

        for _ in range(timed):
            torch.cuda.synchronize()
            start = time.perf_counter()

            model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)

    return {
        "latency_mean_ms": sum(times) / len(times),
        "latency_median_ms": float(torch.tensor(times).median()),
        "latency_p95_ms": float(torch.tensor(times).quantile(0.95)),
        "peak_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "warmup_runs": warmup,
        "timed_runs": timed,
        "batch_size": 1,
        "input_size": "3x224x224",
        "timing_scope": "model forward pass only; preprocessing excluded",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument(
        "--out",
        default="results/tables/inference_profile.csv",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Run this on Colab with an A100 GPU.")

    device = torch.device("cuda")
    checkpoint_root = Path(args.checkpoints_dir)
    rows = []

    print("GPU:", torch.cuda.get_device_name(0))

    for run, epoch, mode, selection_method in CHECKPOINTS:
        print(f"Profiling {run}, epoch {epoch}, method={selection_method}")

        weights = "IMAGENET1K_V1" if mode == "linear_probe" else "none"

        model = build_model(
            num_classes=65,
            mode=mode,
            weights=weights,
        )

        checkpoint = (
            checkpoint_root
            / run
            / f"epoch_{epoch:02d}.pt"
        )

        model = load_checkpoint(model, checkpoint, mode)
        model = model.to(device).eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        result = profile_model(model, device)

        rows.append({
            "run": run,
            "epoch": epoch,
            "selection_method": selection_method,
            "mode": mode,
            "total_params": total_params,
            "trainable_params": trainable_params,
            **result,
        })

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print("\nProfiling results:")
    print(df.to_string(index=False))
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()