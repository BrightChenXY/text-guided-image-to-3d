from pathlib import Path
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Edit these paths for your actual workspace.
RUN_DIR = Path("/workspace/arch1_runs/stage2_poc_500_fresh")
TB_DIR = RUN_DIR / "tb_logs"
OUT = RUN_DIR / "loss_curve_total_smoothed.png"
TAG = "loss/loss"


def moving_average(vals, window=15):
    if not vals:
        return vals
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo:i+1]) / (i - lo + 1))
    return out


def main():
    event_files = sorted(TB_DIR.glob("events.out.tfevents.*"))
    if not event_files:
        raise SystemExit(f"No event files found in {TB_DIR}")

    event_file = str(event_files[-1])
    print("Using:", event_file)

    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print("Available scalar tags:")
    for t in tags:
        print(" -", t)

    if TAG not in tags:
        raise SystemExit(f"Tag {TAG!r} not found. Pick one from the list above.")

    scalars = ea.Scalars(TAG)
    steps = [s.step for s in scalars]
    vals = [s.value for s in scalars]
    smooth = moving_average(vals, window=15)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, vals, alpha=0.35, label="raw")
    plt.plot(steps, smooth, label="smoothed")
    plt.xlabel("Step")
    plt.ylabel(TAG)
    plt.title("Stage 2 total training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT, dpi=200)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
