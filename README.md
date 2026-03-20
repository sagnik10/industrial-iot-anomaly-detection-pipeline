# Industrial IoT Anomaly Detection Pipeline

## Overview

This project implements a complete end-to-end telemetry analytics pipeline for industrial IoT systems. It processes raw sensor data, performs statistical and time-series analysis, detects anomalies using both rule-based and machine learning approaches, and generates a fully automated analytical report with visual insights.

The system is designed to simulate a real-world predictive maintenance workflow where machine sensor data is continuously monitored to identify abnormal behavior and operational inefficiencies.

---

## Key Features

* Data preprocessing and cleaning
* Time-series analysis of sensor signals
* Rule-based anomaly detection using engineering thresholds
* Machine learning anomaly detection using Isolation Forest
* Clustering of operational states using K-Means
* Statistical analysis (rolling mean, variance, cumulative metrics)
* Regression-based trend modeling (linear and polynomial)
* Residual analysis for model evaluation
* Autocorrelation and lag-based temporal analysis
* Automated visualization generation
* Automatic PDF report generation with explanations

---

## Project Structure

```
.
├── analyzer.py
├── dataset1_01.csv
├── output/
│   ├── charts/
│   ├── anomalies/
│   └── Complete_Machine_Analysis_Report.pdf
└── README.md
```

---

## Workflow

1. Load and clean telemetry dataset
2. Normalize and preprocess sensor values
3. Detect anomalies using:

   * Engineering limits (0–500 range)
   * Isolation Forest model
4. Perform statistical analysis:

   * Rolling statistics
   * Cumulative statistics
5. Fit regression models:

   * Linear regression
   * Polynomial regression
6. Perform clustering using K-Means
7. Analyze residuals and temporal dependencies
8. Generate multiple visualizations
9. Compile all insights into a structured PDF report

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* FPDF

---

## Installation

Clone the repository:

```
git clone https://github.com/sagnik10/industrial-iot-anomaly-detection-pipeline.git
cd industrial-iot-anomaly-detection-pipeline
```

Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn fpdf
```

---

## Usage

Run the analysis pipeline:

```
python analyzer.py
```

---

## Output

After execution, the following will be generated:

* Time-series plots with anomaly highlighting
* Statistical charts (mean, variance, distributions)
* Clustering visualizations
* Regression and residual analysis plots
* Autocorrelation and lag plots
* Final PDF report summarizing all insights

Output directory:

```
output/
```

---

## Anomaly Detection Approach

The system combines two approaches:

### Rule-Based Detection

* Values outside the range 0–500 are flagged as anomalies

### Machine Learning Detection

* Isolation Forest identifies outliers based on data distribution

### Final Decision

* A data point is marked anomalous if detected by either method

---

## Machine Learning Components

* Isolation Forest for anomaly detection
* K-Means clustering for operational state segmentation
* Linear Regression for trend analysis
* Polynomial Regression for nonlinear behavior modeling

---

## Use Cases

* Predictive maintenance
* Industrial machine monitoring
* Sensor behavior analysis
* Fault detection in IoT systems
* Time-series anomaly detection research

---

## Future Improvements

* Real-time streaming data integration
* Dashboard visualization (Streamlit or Flask)
* Advanced models (LSTM, ARIMA)
* Multi-sensor correlation analysis
* Deployment as a microservice

---

## License

This project is open-source and available for educational and research purposes.

---

## Author

Sagnik
