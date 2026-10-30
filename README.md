# Optimizing Bike Share Operations Through Data Analytics

This project explores how weather, seasonality, and demand patterns impact daily bike rentals.
Using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset), the workflow covers data corruption simulation, cleaning, feature engineering, and visualization to support bike-share operational decisions.

## Project Overview
- Goal: Identify factors that influence bike rentals and define optimal conditions for operations.
- Dataset: UCI Bike Sharing Dataset (daily records with weather, season, and rental counts).
- Workflow:
1. Corrupt clean source data in controlled ways (missing values, formatting noise, duplicates).
2. Clean and validate corrupted records with a repeatable pipeline.
3. Engineer useful features (weekend indicator, temperature-humidity interaction).
4. Generate visual outputs for insight communication.

## Project Structure
- `script/produce-file.py`: Generates `day_corrupted.csv` from `day.csv` deterministically.
- `src/bike_rental_pipeline.py`: Core pipeline (cleaning, feature engineering, plotting).
- `bike-rental.py`: Thin entrypoint that runs the pipeline.
- `images/`: Saved figures.
- `tests/test_pipeline.py`: Unit tests for core cleaning/feature logic.
- `tests/test_produce_file.py`: Tests that corrupted-data generation is reproducible.

## Presentation
- PPT: `presentation/bike_share_analysis.pptx`
- PDF: `presentation/bike_share_analysis.pdf`

## Requirements
- Python 3.10+
- Packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Reproduce
1. Generate a corrupted dataset from the clean source:

```bash
python script/produce-file.py
```

2. Run the cleaning and analysis pipeline:

```bash
python bike-rental.py
```

Outputs:
- Cleaned dataset: `day_cleaned.csv`
- Figures: `images/correlation_focused.png`, `images/rentals_vs_temp.png`, `images/rentals_by_season.png`, `images/rentals_by_weather.png`

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
