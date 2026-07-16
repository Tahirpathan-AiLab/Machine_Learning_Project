import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("data/interim/data_processed.pkl")
df = df[df["lable"] != "rest"]

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2

df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["lable"] == "bench"]
squat_df = df[df["lable"] == "squat"]
row_df = df[df["lable"] == "row"]
ohp_df = df[df["lable"] == "ohp"]
dead_df = df[df["lable"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()
plt.show()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 100  # Hz

LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]
bench_set["acc_y"].plot()

column = "acc_y"

LowPass.low_pass_filter(
    bench_set,
    col=column,
    sampling_frequency=fs,
    cutoff_frequency=0.4,
    order=10,
)

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(
        dataset,
        col=column,
        sampling_frequency=fs,
        cutoff_frequency=cutoff,
        order=order,
    )

    indexes = argrelextrema(data[f"{column}_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")

    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["lable"].iloc[0].title()
    category = dataset["catagory"].iloc[0].title()
    plt.title(f"{category} ({exercise}): {len(peaks)} Reps")
    plt.show()

    return len(peaks)


count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df["reps"] = df["catagory"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["lable", "catagory", "set"])["reps"].max().reset_index()

for s in df["set"].unique():
    subset = df[df["set"] == s]

    column = "acc_r"
    cutoff = 0.4

    if subset["lable"].iloc[0] == "squat":
        cutoff = 0.35

    if subset["lable"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyr_x"

    if subset["lable"].iloc[0] == "ohp":
        cutoff = 0.35

    reps = count_reps(
        subset,
        cutoff=cutoff,
        column=column,
    )

    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps
    
rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()