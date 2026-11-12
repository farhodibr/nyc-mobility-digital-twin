# NYC Mobility Digital Twin

I started this project with a question. How much can I tell about ride pickups in New York City before the next hour begins? I wanted to see how far past pickup counts could get me, and when weather, events, or location might help. I also wanted to look at a related question. What happens to pickups near a subway disruption when trains actually stop serving a station?

I originally called what I was forecasting ride demand. I hadn't accounted for someone who sees a high fare or a longer wait than they expected and decides against taking the ride. A completed-trip record can't tell me how many people made that choice. I call my target **completed pickups** because that's what I can count.

This is my CUNY SPS MS in Data Science capstone project. I built an hourly forecasting pipeline for NYC taxi zones and a separate causal analysis of measured subway service loss. The July replay is where I put the forecasts, the actual pickup counts, and the timing of those counts together.

## Try the July replay

[Open the live July replay](https://farhodibr.github.io/nyc-mobility-digital-twin/)

Pick a taxi zone, move through July 2026, and compare the original XGBoost forecast with two versions that adjust for recent errors. I only let the replay use actual pickup counts once they would have become available. That matters because a correction that uses a count too early can look better than it would have worked in practice.

From the project folder, run this command and open [localhost:8000](http://localhost:8000) in your browser. The replay assets are included.

```sh
python -m http.server 8000 --directory website/dist
```

The replay covers 738 forecast hours across 262 zones. I ran it retrospectively with a saved model. The [replay notes](website/REPLAY_UPDATE.md) explain the timing rules and how to regenerate the forecasts.

## Follow the notebooks

- **Forecasting.** I start by [building the zone-hour data](notebooks/data_engineering1_organized_v3.ipynb), then [check which weather forecasts were available](notebooks/03_archived_weather_forecast_production_updated.ipynb) under my timing rule. The [data exploration](notebooks/03b_production_dataset_eda.ipynb) leads into the [model comparison](notebooks/04_production_modeling_refactored_v2.ipynb). I use a [separate experiment](notebooks/05_borough_feature_experiment_with_ensemble.ipynb) to see what borough information and an ensemble add.
- **Subway disruptions.** I first [measure service loss](notebooks/03_mta_disruptions_measured_v3_1.ipynb) and [check examples by hand](notebooks/04_mta_disruptions_validation.ipynb). Then I [compare pickup counts](notebooks/08_subway_fhv_causal_analysis_storytelling.ipynb) around disruptions and look at how the results change under different matching choices.

The written reports are in [reports](reports/).

I also put together a [pickup wait baseline](notebooks/09_wait_time_baseline.ipynb) to see how far a zone’s usual wait for that weekday and hour can get me.

## Run the analysis locally

The download script fetches the prepared data and saved model from a fixed Hugging Face revision, about 670 MiB. You can download it without an account. [View the data snapshot](https://huggingface.co/datasets/glavvrach79/nyc-mobility-digital-twin/tree/5bf35b8304f2c43f34946ca070ac3f642c8bf80d)

```sh
conda env create -f environment.yml
conda activate ride-demand
python -m pip install -e . huggingface_hub
python tools/download_data.py
python -m pytest
jupyter lab
```

The prepared snapshot is the starting point for the forecasting, causal analysis, and replay. Rebuilding everything from scratch also takes the original TLC records, archived transit and weather inputs, and credentials for some data collection steps.

## What I checked from a clean copy

I downloaded the public snapshot without signing in and regenerated all 1,160,136 July forecasts with the saved model. I also ran the archived-weather validation and subway validation notebooks. The subway notebook can show the saved labels and summaries from the snapshot. To inspect the underlying train stops for an example, you’ll need the daily reduced train files.

The forecasting EDA and causal analysis notebooks also ran end to end. On Windows, I needed the Conda environment active before launching Jupyter so the plotting libraries could load. The causal notebook’s optional same-zone XGBoost comparison uses a CUDA-capable GPU and takes longer than the main analysis. Set `RUN_XGBOOST_APPENDIX = False` to run just the main analysis and its robustness checks.

I ran the production modeling notebook end to end on a CUDA-capable Windows machine, including the XGBoost, CatBoost, and LightGBM comparisons, feature experiments, and final evaluation. Full training takes substantially longer than running the replay with the saved model.

The borough experiment also ran end to end, including both XGBoost feature configurations, LightGBM training, and the ensemble comparison.

The data engineering and subway measurement notebooks document how I built the inputs. They need source files outside the prepared snapshot.
