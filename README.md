Edge ML Motor Condition Monitoring Pipeline

📌 Overview

This project implements a multi-modal edge data acquisition and predictive maintenance system designed for an Embedded Linux environment (Raspberry Pi 5). It captures synchronized thermal data and high-frequency vibration metrics from industrial electric motors to predict imminent mechanical failures.

Currently, the robust Data Acquisition & Feature Extraction Pipeline is completed. The next phase focuses on deploying lightweight Edge ML anomaly detection models directly on the hardware.

⚙️ Architecture & Data Acquisition

The pipeline communicates directly with hardware peripherals via the I2C bus, employing advanced embedded software concepts:

AMG8833 Thermal Camera: Captures 8x8 thermal grids to detect motor hotspots.

ADXL345 Accelerometer: Captures 3-axis vibration data at high frequencies (400 Hz setup).

Multi-threading & Mutex: Utilizes a dedicated background thread for continuous high-frequency vibration polling, preventing I2C bus collisions via thread locking.

Circular Buffers: Employs deque structures to maintain a rolling window of synchronized telemetry.

Fast Fourier Transform (FFT): Calculates time-domain (RMS, Kurtosis) and frequency-domain (Dominant Frequency, Spectral Energy) features in real-time.

📂 Current Outputs

The motor_multimodal_collector.py script generates two primary outputs per run:

raw_npz/: Contains compressed NumPy tensors (.npz) of raw thermal matrices and vibration windows for future CNN / Deep Learning training.

features_*.csv: A tabular dataset containing real-time extracted statistical features ready for scikit-learn models.

🚀 To-Do / Next Steps (WIP)

[x] Establish I2C sensor communication and multi-threaded data logging.

[x] Implement real-time FFT and statistical feature extraction.

[ ] Train an Isolation Forest anomaly detection model using the generated .csv dataset.

[ ] Integrate real-time ML inference into the pipeline for live fault alerting.

Developed by Yavuzhan Melek as part of an edge computing and predictive maintenance portfolio.
