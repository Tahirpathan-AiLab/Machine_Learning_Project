## Code Description

This project focuses on transforming cleaned wearable sensor data into meaningful machine learning features that can accurately represent different barbell exercises performed by athletes. Using accelerometer and gyroscope measurements collected from **MetaMotion wearable fitness tracking bands**, the project builds a complete feature engineering pipeline capable of converting raw time-series signals into informative statistical, temporal, frequency-domain, and clustering features.

The original MetaMotion dataset was collected by **Dave Ebbelaar** and his collaborators during workout sessions involving **five participants** performing multiple barbell exercises under different weight conditions. The exercises include **Bench Press, Squat, Overhead Press, Deadlift, Row, and Rest**, while each exercise was performed using both **Heavy (5 repetitions)** and **Medium (10 repetitions)** weight categories.

My contribution begins after the dataset has been cleaned through preprocessing and statistical outlier removal. In this project, I designed and implemented the complete feature engineering pipeline using Python, preparing the dataset for machine learning tasks such as **exercise classification**, **human activity recognition**, and **automatic repetition counting**.

Unlike simple preprocessing, feature engineering aims to discover hidden patterns within sensor signals. Instead of using only the original accelerometer and gyroscope values, this project extracts multiple representations of the same movement, allowing machine learning models to better understand both the physical motion and the temporal behavior of each exercise.

The repository demonstrates how wearable sensor recordings can be transformed into highly informative machine learning features by combining digital signal processing, statistical analysis, temporal abstraction, frequency-domain analysis, dimensionality reduction, vector magnitude computation, and unsupervised clustering techniques.

---

## Feature Engineering Pipeline

The workflow begins by loading the cleaned dataset generated during the preprocessing stage. Since Chauvenet's Criterion replaces detected outliers with missing values (NaN), the first step is to recover these values through interpolation. Linear interpolation estimates the missing observations from neighboring sensor values while preserving the continuity of the original movement signals.

Once the dataset contains no missing values, the duration of every workout set is calculated. The project measures the time difference between the first and last timestamp of each exercise set, providing additional contextual information regarding workout intensity and exercise duration. Average durations are also computed for different weight categories, helping distinguish between heavy and medium repetitions.

To remove high-frequency sensor noise introduced by body vibrations and hardware limitations, a **Butterworth Low-Pass Filter** is applied to each accelerometer and gyroscope axis. The Butterworth filter smooths the sensor signals while preserving the important movement patterns required for exercise recognition. This significantly improves signal quality before extracting higher-level features.

The project then performs **Principal Component Analysis (PCA)** to reduce redundancy between highly correlated sensor axes. Since accelerometer and gyroscope measurements often capture similar information across multiple dimensions, PCA transforms the six original sensor variables into a smaller number of orthogonal principal components while preserving most of the original variance. This reduces feature dimensionality, minimizes multicollinearity, and improves computational efficiency.

To better represent body movement independent of device orientation, vector magnitude features are computed for both the accelerometer and gyroscope. Rather than analyzing each axis separately, the Euclidean magnitude combines all three axes into a single motion intensity measurement.

The following equations are implemented:

**Accelerometer Magnitude**

\[
Acc_r = \sqrt{acc_x^2 + acc_y^2 + acc_z^2}
\]

**Gyroscope Magnitude**

\[
Gyr_r = \sqrt{gyr_x^2 + gyr_y^2 + gyr_z^2}
\]

These magnitude features provide a more stable representation of overall movement intensity during each exercise.

The next stage performs **Temporal Abstraction**, where rolling statistical windows are applied across every sensor signal. Instead of considering only the current observation, the algorithm summarizes recent movement history using rolling mean and rolling standard deviation. These temporal statistics capture trends, motion consistency, and short-term variations that are highly informative for activity recognition.

Because physical exercises exhibit repetitive motion patterns, the project also extracts **Frequency-Domain Features** using the **Fast Fourier Transform (FFT)**. The Fourier Transformation converts time-domain sensor signals into the frequency domain, allowing the pipeline to identify dominant movement frequencies generated during repetitive exercises.

For every rolling window, several frequency features are calculated, including:

- Dominant Frequency
- Frequency Weighted Average
- Power Spectral Entropy (PSE)
- Individual FFT amplitudes across multiple frequency bins

These features capture periodic movement characteristics that are often invisible in the original sensor recordings.

Finally, the project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to group similar movement patterns without using exercise labels. By analyzing only accelerometer measurements, K-Means discovers natural clusters corresponding to similar body movements. The elbow method is used to determine the optimal number of clusters before assigning every observation to its nearest centroid.

The final output is a comprehensive machine learning dataset containing original sensor values, statistical features, temporal features, frequency-domain features, principal components, vector magnitude features, clustering labels, and workout metadata. This enriched dataset provides a strong foundation for supervised learning algorithms capable of accurately recognizing exercises and estimating repetition counts.

---

## Methods Implemented

This project combines multiple signal processing and machine learning techniques to maximize the quality of extracted features.

### Missing Value Imputation

Missing values introduced during statistical outlier removal are reconstructed using linear interpolation, preserving the continuity of time-series sensor signals while avoiding unnecessary information loss.

### Butterworth Low-Pass Filter

The Butterworth filter removes high-frequency noise while preserving important exercise movement patterns. This smoothing process improves feature quality and increases the stability of downstream algorithms.

### Principal Component Analysis (PCA)

Principal Component Analysis transforms correlated sensor measurements into a smaller set of independent principal components while retaining most of the information contained within the original dataset.

### Vector Magnitude Features

Acceleration and angular velocity magnitudes are calculated using the Euclidean norm to represent overall body movement regardless of sensor orientation.

### Temporal Abstraction

Rolling statistical windows compute local mean and standard deviation values, allowing machine learning algorithms to understand recent movement history instead of relying on individual sensor observations.

### Fourier Transformation (FFT)

Fast Fourier Transformation converts time-domain sensor data into frequency-domain representations, making repetitive movement frequencies measurable and suitable for machine learning.

### K-Means Clustering

K-Means clustering groups similar movement patterns by minimizing intra-cluster distances. The elbow method is used to estimate the optimal number of clusters before assigning cluster labels to the dataset.

### Feature Export

The final engineered dataset is exported as a Pickle file, allowing future machine learning models to directly use the generated features without repeating the complete feature engineering pipeline.

---

## Key Features

* Loads the cleaned dataset generated after statistical outlier removal
* Handles missing values using linear interpolation
* Calculates workout duration for every exercise set
* Computes average duration for different weight categories
* Applies a Butterworth Low-Pass Filter to remove high-frequency sensor noise
* Smooths accelerometer and gyroscope signals while preserving movement patterns
* Performs Principal Component Analysis (PCA) for dimensionality reduction
* Generates principal component features while retaining maximum variance
* Calculates vector magnitude features for both accelerometer and gyroscope sensors
* Performs Temporal Abstraction using rolling statistical windows
* Extracts rolling mean and rolling standard deviation features
* Applies Fast Fourier Transformation (FFT) for frequency-domain feature extraction
* Computes dominant frequency, weighted frequency, and Power Spectral Entropy (PSE)
* Generates FFT amplitudes across multiple frequency bins
* Uses K-Means clustering to discover similar movement patterns
* Determines the optimal number of clusters using the Elbow Method
* Visualizes intermediate processing steps throughout the pipeline
* Produces a comprehensive feature-rich dataset ready for machine learning
* Exports the final engineered dataset as a reusable Pickle file

---

## Workflow

<p align="center">
  <img src="https://img.shields.io/badge/1-Import%20Libraries-4CAF50?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/2-Load%20Cleaned%20Dataset-2196F3?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/3-Handle%20Missing%20Values-FF9800?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/4-Calculate%20Workout%20Duration-E91E63?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/5-Butterworth%20Low--Pass%20Filtering-9C27B0?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/6-Principal%20Component%20Analysis-00BCD4?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/7-Generate%20Vector%20Magnitude-795548?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/8-Temporal%20Feature%20Extraction-607D8B?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/9-Frequency%20Feature%20Extraction-3F51B5?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/10-K--Means%20Clustering-009688?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/11-Generate%20Feature%20Dataset-8BC34A?style=for-the-badge"/>
</p>

<p align="center">⬇️</p>

<p align="center">
  <img src="https://img.shields.io/badge/12-Export%20Features-4CAF50?style=for-the-badge"/>
</p>

---

## Technologies Used

<p align="center">
  <img src="https://skillicons.dev/icons?i=python" alt="Python"/>
  <img src="https://skillicons.dev/icons?i=vscode" alt="VS Code"/>
  <img src="https://skillicons.dev/icons?i=git" alt="Git"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Fast%20Fourier%20Transform-673AB7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Butterworth%20Filter-009688?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Pickle-FFCA28?style=for-the-badge"/>
</p>

---

## Project Structure

```text
Machine_Learning_Project
│
├── data
│   ├── raw
│   │   └── MetaMotion
│   │       ├── Accelerometer CSV Files
│   │       └── Gyroscope CSV Files
│   │
│   └── interim
│       ├── data_processed.pkl
│       ├── outliers_removed_chauvenet.pkl
│       └── data_features.pkl
│
├── src
│   ├── features
│   │   ├── build_features.py
│   │   ├── remove_outliers.py
│   │   ├── count_repetitions.py
│   │   ├── DataTransformation.py
│   │   ├── TemporalAbstraction.py
│   │   └── FrequencyAbstraction.py
│   │
│   └── models
│       └── ...
│
├── requirements.txt
└── README.md
```

---

## Generated Features

The final dataset contains a rich collection of engineered features extracted from the original accelerometer and gyroscope recordings.

### Original Sensor Features

- acc_x
- acc_y
- acc_z
- gyr_x
- gyr_y
- gyr_z

### Principal Components

- pca_1
- pca_2
- pca_3

### Vector Magnitude Features

- acc_r
- gyr_r

### Temporal Features

Rolling Mean

- acc_x_temp_mean
- acc_y_temp_mean
- acc_z_temp_mean
- gyr_x_temp_mean
- gyr_y_temp_mean
- gyr_z_temp_mean
- acc_r_temp_mean
- gyr_r_temp_mean

Rolling Standard Deviation

- acc_x_temp_std
- acc_y_temp_std
- acc_z_temp_std
- gyr_x_temp_std
- gyr_y_temp_std
- gyr_z_temp_std
- acc_r_temp_std
- gyr_r_temp_std

### Frequency Features

For every predictor column the following features are generated:

- Maximum Frequency
- Weighted Frequency
- Power Spectral Entropy (PSE)
- FFT amplitudes across multiple frequency bins

### Clustering Features

- Cluster ID generated using K-Means

---

## Output

After successful execution, the project generates:

* **data_features.pkl** – Complete feature engineered dataset ready for machine learning.
* Smoothed accelerometer and gyroscope signals after Butterworth filtering.
* Principal Component Analysis (PCA) features.
* Accelerometer and gyroscope vector magnitude features.
* Rolling mean and rolling standard deviation features.
* Frequency-domain features generated using Fast Fourier Transformation.
* Dominant frequency and Power Spectral Entropy measurements.
* K-Means cluster labels for every observation.
* Multiple visualization plots demonstrating each feature engineering stage.
* A reusable dataset suitable for classification, clustering, and repetition counting.

---

## Applications

The generated feature dataset can directly be used for several wearable AI and machine learning applications, including:

* Barbell Exercise Classification
* Automatic Repetition Counting
* Human Activity Recognition (HAR)
* Workout Monitoring
* Athlete Performance Analysis
* Smart Fitness Assistants
* Wearable AI Applications
* Movement Pattern Recognition
* Exercise Recommendation Systems
* Sports Analytics

---

## Acknowledgements

The original MetaMotion wearable sensor dataset used in this project was collected by **Dave Ebbelaar** and his collaborators as part of the **Machine Learning for the Quantified Self** learning materials inspired by the work of **Mark Hoogendoorn** and **Burkhardt Funk**.

This repository focuses on the complete feature engineering pipeline built on top of the cleaned dataset. All preprocessing, signal transformation, temporal abstraction, frequency analysis, clustering, visualization, and feature generation workflows were implemented in Python to prepare high-quality inputs for machine learning models.

---

## Conclusion

This project represents the **Feature Engineering** stage of my Machine Learning pipeline. It builds upon the cleaned sensor dataset by transforming raw accelerometer and gyroscope measurements into meaningful numerical representations using modern signal processing, statistical analysis, dimensionality reduction, temporal abstraction, frequency-domain analysis, and unsupervised learning techniques.

The resulting feature-rich dataset captures both spatial and temporal characteristics of human movement, making it highly suitable for machine learning algorithms that perform exercise classification, repetition counting, and human activity recognition. By combining multiple feature extraction techniques into a single reproducible workflow, the project provides a scalable foundation for developing intelligent wearable fitness applications and real-time activity recognition systems.
