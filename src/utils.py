import os
import pandas as pd
import matplotlib.pyplot as plt


def save_results(res_dict, out_dir, name):
    """
    Save metrics to CSV, save plot as PNG, and try to display the plot.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Save metrics to CSV ----
    df = pd.DataFrame([res_dict])
    csv_path = os.path.join(out_dir, f"{name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV for {name} -> {csv_path}")

    # ---- Save + show plot ----
    try:
        print(f"Saving + displaying plot for {name}...")

        plt.figure(figsize=(4, 3))
        plt.bar([name], [res_dict["avg_response_time"]])
        plt.ylabel("Avg Response Time")
        plt.title(f"Avg Response Time - {name}")
        plt.tight_layout()

        img_path = os.path.join(out_dir, f"{name}_avg_response.png")
        plt.savefig(img_path)
        print(f"Saved PNG for {name} -> {img_path}")

        # show the plot in a window
        plt.show()          # this should open a GUI window
        plt.close()
    except Exception as e:
        print("Plotting failed:", e)
