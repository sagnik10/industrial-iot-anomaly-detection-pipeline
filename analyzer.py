import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,silhouette_score
from pandas.plotting import autocorrelation_plot

base_dir=os.path.dirname(os.path.abspath(__file__))
data_path=os.path.join(base_dir,"dataset1_01.csv")

data=pd.read_csv(data_path)

output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")
anomaly_dir=os.path.join(output_dir,"anomalies")

os.makedirs(output_dir,exist_ok=True)
os.makedirs(charts_dir,exist_ok=True)
os.makedirs(anomaly_dir,exist_ok=True)

data.columns=data.columns.str.strip().str.replace(" ","_")
data["log_time"]=data["log_time"].astype(str).str.replace('="','').str.replace('"','')
data["log_time"]=pd.to_datetime(data["log_time"],errors="coerce")

data=data.drop_duplicates()
data=data.dropna(subset=["log_time"])
data=data.sort_values("log_time")

numeric_cols=["int_value","scalar_value","raw_value"]

for col in numeric_cols:
    data[col]=pd.to_numeric(data[col],errors="coerce")

data[numeric_cols]=data[numeric_cols].ffill()

machines=data["engine_details"].unique()
sensor_units=data.groupby("sensor_name")["unit"].first().to_dict()

chart_paths=[]
machine_accuracy={}

for machine in machines:

    mdata=data[data["engine_details"]==machine]
    sensors=mdata["sensor_name"].unique()

    machine_accuracy_values=[]

    for sensor in sensors:

        sdata=mdata[mdata["sensor_name"]==sensor].copy()

        unit=sensor_units.get(sensor,"units")

        sdata["limit_anomaly"]=((sdata["scalar_value"]<0)|(sdata["scalar_value"]>500)).astype(int)

        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(sdata[["scalar_value"]])

        model=IsolationForest(contamination=0.05,random_state=42)
        model.fit(X_scaled)

        sdata["iso_anomaly"]=np.where(model.predict(X_scaled)==-1,1,0)
        sdata["final_anomaly"]=((sdata["limit_anomaly"]==1)|(sdata["iso_anomaly"]==1)).astype(int)

        anomalies=sdata[sdata["final_anomaly"]==1]

        rolling_mean=sdata["scalar_value"].rolling(10).mean()
        rolling_std=sdata["scalar_value"].rolling(10).std()
        rolling_median=sdata["scalar_value"].rolling(10).median()
        rolling_var=sdata["scalar_value"].rolling(10).var()

        cumulative_mean=sdata["scalar_value"].expanding().mean()
        cumulative_var=sdata["scalar_value"].expanding().var()

        x=np.arange(len(sdata)).reshape(-1,1)
        poly=PolynomialFeatures(2)
        xp=poly.fit_transform(x)

        poly_model=LinearRegression()
        poly_model.fit(xp,sdata["scalar_value"])
        poly_trend=poly_model.predict(xp)

        lin_model=LinearRegression()
        lin_model.fit(x,sdata["scalar_value"])
        lin_trend=lin_model.predict(x)

        residuals=sdata["scalar_value"]-poly_trend

        kmeans=KMeans(n_clusters=3,random_state=42)
        clusters=kmeans.fit_predict(X_scaled)

        sil=silhouette_score(X_scaled,clusters)

        def save(title,explanation):
            path=os.path.join(charts_dir,title.replace(" ","_")+".png")
            plt.savefig(path)
            plt.close()
            chart_paths.append((title,path,explanation))

        plt.figure(figsize=(11,5))
        plt.plot(sdata["log_time"],sdata["scalar_value"],label="Sensor value")
        plt.scatter(anomalies["log_time"],anomalies["scalar_value"],color="red",label="Anomaly")
        plt.xlabel("Timestamp")
        plt.ylabel(f"Sensor value ({unit})")
        plt.title(f"{sensor} Time Series Behaviour")
        plt.legend()
        save(f"{sensor} Time Series",
f"""
This time-series chart shows how the sensor measurement evolves over chronological time.

The horizontal axis represents timestamps recorded by the telemetry system.
The vertical axis represents sensor values measured in {unit}.

Engineering constraints define the acceptable operational range between 0 and 500.
Values above this threshold are automatically classified as anomalies.

Red markers indicate detected anomalies combining engineering rules and machine learning detection.

Total observations: {len(sdata)}
Detected anomalies: {len(anomalies)}
Silhouette cluster score: {round(sil,4)}
""")

        def stat_chart(series,title,xlab,ylab,text):
            plt.figure(figsize=(10,5))
            plt.plot(series)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
            save(title,text)

        stat_chart(rolling_mean,f"{sensor} Rolling Mean","Observation Index",f"Rolling Mean ({unit})",
f"""
Rolling mean smoothing reveals the long-term behaviour of the telemetry signal.
Short-term noise is reduced allowing engineers to observe structural trends
in the machine operation.
""")

        stat_chart(rolling_std,f"{sensor} Rolling Standard Deviation","Observation Index",f"Std ({unit})",
"""
Rolling standard deviation represents signal volatility.
Higher values indicate unstable or fluctuating machine behaviour.
""")

        stat_chart(rolling_median,f"{sensor} Rolling Median","Observation Index",f"Median ({unit})",
"""
Rolling median is robust against outliers and provides a stable
representation of the central tendency of the signal.
""")

        stat_chart(rolling_var,f"{sensor} Rolling Variance","Observation Index",f"Variance ({unit})",
"""
Rolling variance measures variability in the telemetry signal.
Increasing variance may indicate unstable machine conditions.
""")

        stat_chart(cumulative_mean,f"{sensor} Cumulative Mean","Observation Index",f"Mean ({unit})",
"""
Cumulative mean tracks the average value of the signal as more
observations are accumulated over time.
""")

        stat_chart(cumulative_var,f"{sensor} Cumulative Variance","Observation Index",f"Variance ({unit})",
"""
Cumulative variance illustrates how variability evolves
as more telemetry observations are collected.
""")

        plt.figure(figsize=(10,5))
        plt.plot(poly_trend,label="Polynomial trend")
        plt.plot(lin_trend,label="Linear trend")
        plt.xlabel("Observation Index")
        plt.ylabel(f"Sensor value ({unit})")
        plt.title(f"{sensor} Regression Trends")
        plt.legend()
        save(f"{sensor} Regression Trends",
"""
Polynomial and linear regression models approximate long-term behaviour
of the sensor signal.

Polynomial regression captures nonlinear drift patterns that may occur
due to thermal expansion, mechanical wear or operational load changes.
""")

        def dist_plot(func,title,explanation):
            plt.figure(figsize=(7,5))
            func()
            plt.xlabel(f"Sensor value ({unit})")
            plt.ylabel("Frequency / Density")
            plt.title(title)
            save(title,explanation)

        dist_plot(lambda: sns.histplot(sdata["scalar_value"],bins=30,kde=True),
f"{sensor} Histogram",
"""
Histogram visualizes the statistical distribution of sensor values.
The shape of the distribution reveals the most common operating ranges
of the machine.
""")

        dist_plot(lambda: sns.kdeplot(sdata["scalar_value"]),
f"{sensor} Density Distribution",
"""
Kernel density estimation approximates the probability distribution
of the telemetry signal.
""")

        dist_plot(lambda: sns.boxplot(y=sdata["scalar_value"]),
f"{sensor} Boxplot",
"""
Boxplot displays quartiles, median and extreme outliers in the data.
""")

        dist_plot(lambda: sns.violinplot(y=sdata["scalar_value"]),
f"{sensor} Violin Plot",
"""
Violin plots combine density estimation and quartile information
to represent the full distribution of sensor values.
""")

        plt.figure(figsize=(7,5))
        sns.scatterplot(x=sdata["scalar_value"],y=clusters)
        plt.xlabel(f"Sensor value ({unit})")
        plt.ylabel("Cluster label")
        plt.title(f"{sensor} Cluster Behaviour Map")
        save(f"{sensor} Cluster Map",
"""
K-Means clustering partitions the telemetry data into behavioural regimes
representing potential machine operating states.
""")

        plt.figure(figsize=(7,5))
        sns.histplot(clusters)
        plt.xlabel("Cluster")
        plt.ylabel("Frequency")
        plt.title(f"{sensor} Cluster Distribution")
        save(f"{sensor} Cluster Distribution",
"""
Cluster distribution shows how frequently each operational regime occurs.
""")

        plt.figure(figsize=(7,5))
        plt.hist(residuals,bins=30)
        plt.xlabel("Residual error")
        plt.ylabel("Frequency")
        plt.title(f"{sensor} Residual Distribution")
        save(f"{sensor} Residual Histogram",
"""
Residual histogram illustrates the prediction error distribution
between the regression model and the observed data.
""")

        plt.figure(figsize=(7,5))
        plt.scatter(poly_trend,residuals)
        plt.xlabel("Predicted values")
        plt.ylabel("Residual error")
        plt.title(f"{sensor} Residual Scatter")
        save(f"{sensor} Residual Scatter",
"""
Residual scatter plots show the distribution of prediction errors.
""")

        plt.figure(figsize=(7,5))
        autocorrelation_plot(sdata["scalar_value"])
        plt.title(f"{sensor} Autocorrelation")
        save(f"{sensor} Autocorrelation",
"""
Autocorrelation analysis reveals temporal relationships
between sequential observations.
""")

        for i in range(1,8):
            lag=sdata["scalar_value"].shift(i)
            plt.figure(figsize=(7,5))
            plt.scatter(lag,sdata["scalar_value"])
            plt.xlabel(f"Lag {i}")
            plt.ylabel("Current value")
            plt.title(f"{sensor} Lag Plot {i}")
            save(f"{sensor} Lag Plot {i}",
"""
Lag plots reveal time-dependent relationships
between consecutive sensor observations.
""")

        accuracy=1-(abs(sdata["raw_value"]-sdata["int_value"])/sdata["raw_value"].max())
        machine_accuracy_values.append(accuracy.mean())

    machine_accuracy[machine]=np.mean(machine_accuracy_values)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)

pdf.add_page()
pdf.set_font("Arial","B",20)
pdf.cell(0,12,"Machine Telemetry Analytical Report",0,1,'C')

intro="""
This report analyses machine telemetry behaviour using engineering
constraints and machine learning techniques.

The MEM.speed sensor operates within an acceptable range between 0 and 500.
Values exceeding this threshold are classified as anomalies.

Statistical and machine learning methods are applied to identify abnormal
machine behaviour and understand sensor dynamics.
"""

pdf.set_font("Arial","",12)
pdf.multi_cell(0,8,intro)

for title,path,explanation in chart_paths:

    pdf.add_page()
    pdf.set_font("Arial","B",14)
    pdf.cell(0,8,title,0,1)
    pdf.image(path,w=180)
    pdf.ln(5)
    pdf.set_font("Arial","",11)
    pdf.multi_cell(0,7,explanation)

pdf.output(os.path.join(output_dir,"Complete_Machine_Analysis_Report.pdf"))

print("All outputs saved in:",output_dir)