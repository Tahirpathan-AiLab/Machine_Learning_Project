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
