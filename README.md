# Optimizing Bike Share Operations Through Data Analytics

This project explores how weather, seasonality, and demand patterns impact daily bike rentals.  
Using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset), I applied data cleaning, exploratory data analysis (EDA), and visualization to uncover insights that can guide operational decision-making for bike share programs.

---

# Project Overview
- **Goal:** Identify factors that influence bike rentals and define optimal conditions for operations.
- **Dataset:** UCI Bike Sharing Dataset (daily records: weather, season, and rental counts).
- **Key Steps:**
	1. Clean corrupted dataset (duplicates, NaNs, outliers, invalid strings).
	2. Explore relationships between features (temperature, humidity, weather situation).
	3. Visualize findings with Python (matplotlib, seaborn).
	4. Highlight operational insights (e.g., "Goldilocks zone" for temperature).

---

## Data Cleaning

Real-world datasets are often messy. To simulate this, I introduced errors such as missing values, duplicates, and inconsistent formats using a small helper script (`script/produce-file.py`).  
These were then cleaned and validated in the main analysis script.

This demonstrates both:
- Identifying and fixing data quality issues
- Ensuring the analysis pipeline is robust

---

## Presentation
A detailed walkthrough of the analysis and recommendations as well as the graphs produced by my code is available here:  
👉 [Optimizing Bike Share Operations Through Data Analytics (PPT)](presentation/bike_share_analysis.pptx)


*(PDF version available for easy viewing: [link](presentation/bike_share_analysis.pdf))*

---

## Tools & Libraries
- **Python** (pandas, numpy, matplotlib, seaborn)
- **PowerPoint** for communicating findings

---

## How to Reproduce
1. Clone this repository:
   ```bash
   git clone https://github.com/AbbaAdam1/bike-rental-project
   cd bike-rental-project
