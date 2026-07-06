import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    "data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("data/raw/MetaMotion/*.csv")

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
f = files[10]

participants = f.split("-")[0].replace("data/raw/MetaMotion\\", "")
lable = f.split("-")[1]
catagory = f.split("-")[2].rstrip("12345").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participants"] = participants
df["lable"] = lable
df["catagory"] = catagory


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participants = f.split("-")[0].replace("data/raw/MetaMotion\\", "")
    lable = f.split("-")[1]
    catagory = f.split("-")[2].rstrip("12345").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participants"] = participants
    df["lable"] = lable
    df["catagory"] = catagory

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_df = pd.concat([acc_df, df])
        acc_set += 1

    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_df = pd.concat([gyr_df, df])
        gyr_set += 1


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()

    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participants = f.split("-")[0].replace("data/raw/MetaMotion\\", "")
        lable = f.split("-")[1]
        catagory = f.split("-")[2].rstrip("12345").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participants"] = participants
        df["lable"] = lable
        df["catagory"] = catagory

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_df = pd.concat([acc_df, df])
            acc_set += 1

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_df = pd.concat([gyr_df, df])
            gyr_set += 1

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")

    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merge = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

data_merge.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participants",
    "lable",
    "catagory",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participants": "last",
    "lable": "last",
    "catagory": "last",
    "set": "last",
}

data_merge.resample(rule="200ms").apply(sampling)

days = [ g for n, g in data_merge.groupby(pd.Grouper(freq="D")) ]

data_resample = pd.concat(df.resample(rule="200ms").apply(sampling).dropna() for df in days)
data_resample["set"] = data_resample["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resample.to_pickle("data/interim/data_processed.pkl")