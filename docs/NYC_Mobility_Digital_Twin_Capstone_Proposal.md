# NYC Mobility Digital Twin and Ride Demand Intelligence Platform

**Capstone Project Proposal** \| Farhod Ibragimov

# Section 1 Introduction

New York City ride demand changes sharply across hours and neighborhoods. Weather, commuting schedules, sporting events, concerts, and subway disruptions can all change when and where passengers request high-volume for-hire vehicle services such as Uber and Lyft. I plan to build a system that predicts the number of pickups in every NYC taxi zone for each of the next one through six hours. The practical goal is simple: provide an early warning of where demand may increase so transportation operators can prepare before passengers experience long waits or vehicles remain concentrated in the wrong locations.

The problem is difficult because NYC contains neighborhoods with very different demand levels, and the same neighborhood can behave differently by hour, weekday, weather condition, or event schedule. Large demand surges are uncommon, which makes them harder to learn. Information also becomes available at different times. A useful forecast cannot rely on weather observations, event outcomes, or trip counts that would not yet be known when the forecast is issued.

I will address two connected questions. The first is predictive: how accurately can I forecast hourly ride demand by taxi zone while using only information available at the forecast time? The second is causal: when subway service is disrupted, does ride-hailing activity change in affected origin and destination areas? I will test whether pickups increase near disrupted stations and whether drop-offs increase inside subway outage areas. More drop-offs may mean that people who cannot reach those destinations by subway switch to ride-hailing for the remaining part of their trip. I will also examine whether the effect spreads directionally from disrupted stations toward the areas passengers were trying to reach. The disruption analysis will inform future forecasting features, but I will not treat a predictive association as proof of a causal effect.

## Objectives and research questions

- Build a reproducible pipeline that converts trip records, weather forecasts, calendars, and scheduled events into hourly zone-level forecasts for horizons one through six.

- Compare a weekly seasonal forecast with machine-learning models and determine whether improvements persist across horizons, boroughs, activity levels, and high-demand periods.

- Investigate whether remaining forecast errors show stable differences between zones and stable patterns within zones before applying a hierarchical residual correction.

- Produce prediction intervals so the platform communicates uncertainty rather than presenting every point forecast as equally reliable.

- Estimate whether subway disruptions cause changes in ride-hailing pickups and drop-offs near affected stations and along plausible travel directions.

- Deploy an interactive dashboard showing forecasts, intervals, explanations, data freshness, and the source forecast run used for each prediction.

**Who cares and what could change.** Transportation operators, passengers, city planners, and emergency-management teams could use these results. If successful, the project could support earlier vehicle positioning, event-release planning, monitoring during severe weather or transit disruptions, and clearer communication about where forecasts are uncertain.

## How the proposed project differs from existing work

Taxi-demand research is often strongest in model design but weaker in documenting whether every input would have been available at prediction time. Public portfolio projects often provide attractive applications but use a short data period or simulated demand. Many works focus mainly on model accuracy, but does not show the full path from raw data to a forecast that can be reproduced. My project will combine research-level evaluation with a complete operational data pipeline.

| **Prior work**                                                     | **Main approach**                                                       | **Their advantage**                          | **Planned advantage here**                                             |
|--------------------------------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------|
| [<u>DMVST-Net</u>](https://arxiv.org/abs/1802.08714)               | LSTM, local CNN, and similarity-based spatial views.                    | Stronger explicit spatial learning.          | Forecast-time data lineage, six horizons, and production checks.       |
| [<u>ST-ResNet</u>](https://arxiv.org/abs/1610.00081)               | Recent, daily, and long-term temporal components with external factors. | Stronger deep spatial-temporal architecture. | TLC-zone outputs and realistic archived weather availability.          |
| [<u>Generalized STAR</u>](https://arxiv.org/abs/1711.10090)        | Interpretable spatial autoregression for NYC yellow taxi demand.        | Clear neighboring-region effects.            | Larger feature pipeline, event data, and operational backtesting.      |
| [<u>Event-text model</u>](https://arxiv.org/abs/1808.05535)        | Combines taxi time series with descriptions of NYC events.              | Richer event representation.                 | Broader city coverage and controlled event-window comparisons.         |
| [<u>Real-time taxi dispatch</u>](https://arxiv.org/abs/1603.04418) | Connects demand estimates to vehicle-dispatch optimization.             | Directly evaluates an operational decision.  | More detailed NYC demand and external-data pipeline.                   |
| [<u>Multi-source attention</u>](https://arxiv.org/abs/2005.05128)  | Combines recurrent networks, attention, weather, and event features.    | Stronger deep temporal architecture.         | Stricter feature-availability and chronological validation rules.      |
| [<u>RideCastAI</u>](https://github.com/rajesh1804/RideCastAI)      | A deployed fare and ETA application with quantile estimates.            | Public interface and latency display.        | Actual future zone-demand forecasting rather than a simulated heatmap. |

## Additional differences in my proposed approach

| **Dimension**           | **Common limitation**                                                                | **How this project will differ**                                                                     |
|-------------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Geographic coverage     | Some studies use grids, clusters, or selected event areas.                           | I will forecast all 263 TLC zones and retain sparse zones instead of removing difficult locations.   |
| Complete demand grid    | Trip-only tables can omit hours with no recorded pickups.                            | I will create every zone-hour combination, then distinguish real zero demand from missing data.      |
| Forecast-time weather   | Observed target-hour weather is often used during historical testing.                | I will select the archived ECMWF run available at issuance, including its assumed reporting delay.   |
| Multi-horizon testing   | Many projects predict one next period or use a random train-test split.              | Every origin will predict hours 1-6, with boundary origins purged so target windows cannot overlap.  |
| Feature evidence        | Weather and event variables are often included without testing their specific value. | I will compare event windows and feature groups against controlled versions of the same model.       |
| Remaining error         | A single citywide score can hide repeated errors in particular places and times.     | I will examine between-zone and within-zone residual variation before accepting partial pooling.     |
| Uncertainty and lineage | Point forecasts and polished maps may hide how uncertain or stale a result is.       | The dashboard will show prediction intervals, data freshness, and the exact weather run used.        |
| Subway disruptions      | Demand papers usually treat disruptions as another predictor or do not include them. | I will separately estimate directional pickup and drop-off effects with a causal event-study design. |

**Main difference.** This make the project different mainly because the model will be evaluated under the same information limits that a real hourly forecast would face. A complex method will be kept only when it improves later chronological data. The subway study will remain a separate causal analysis because a forecast model cannot tell what would have happened without the disruption.

# Section 2 Data

I plan to integrate several public and archived sources at an hourly geographic level. The main analytical key will be taxi zone and hour, while the forecasting table will additionally include the time when a forecast is issued and the forecast horizon. This distinction will allow me to check whether each value was available before prediction.

| **Dataset**                        | **Planned content**                                                                                                  | **Role in the project**                                                                                                               |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| NYC TLC HVFHV trip records         | 2021 through the most recent complete month; pickup and drop-off timestamps and taxi-zone identifiers.               | Construct hourly pickups, drop-offs, lagged demand, and validation outcomes.                                                          |
| NYC TLC taxi-zone geography        | Polygon boundaries, borough labels, and zone identifiers.                                                            | Build a complete 263-zone grid, map stations and venues, and summarize geographic errors.                                             |
| Open-Meteo ECMWF IFS archive       | Historical forecast runs, temperatures, precipitation, snowfall, rain, and wind.                                     | Use the weather forecast that would have been available at each issuance time instead of observed future weather.                     |
| ESPN schedules                     | Historical home games for major NYC basketball, hockey, and baseball teams.                                          | Create venue-specific scheduled and post-game demand windows.                                                                         |
| Setlist.fm                         | Historical concert records for selected large NYC venues.                                                            | Create concert windows and later add event category or popularity information.                                                        |
| MTA service alerts and schedules   | Alert updates, affected routes or stations, status, text, and timestamps; scheduled service changes where available. | Identify disruption start, evolution, resolution, severity, affected geography, and whether the event was knowable before a forecast. |
| Subway station and route geography | Station coordinates, served routes, and links between stations.                                                      | Assign disruptions to taxi zones and define affected origins, destinations, nearby comparison areas, and directional spillovers.      |

## Planned processing and integration

I will preserve raw downloads before producing cleaned silver and modeling-ready gold tables. Trip records will be aggregated into a complete hourly grid so an absent trip record becomes zero demand only after the corresponding zone-hour is known to exist. Duplicate keys, invalid zones, missing hours, time-zone handling, and daylight-saving transitions will be tested explicitly. Weather and event sources will be cached and joined by time and location.

For forecasting, each row will represent a forecast origin, a horizon from one through six, and a taxi zone. Demand lags will be calculated on the uninterrupted historical series before applying chronological splits. Archived weather runs will be selected only when their assumed availability time is earlier than the forecast origin. The train, validation, and test periods will be separated chronologically, with boundary origins removed so their six-hour target windows cannot overlap.

For subway disruptions, alert text and affected-route fields will be converted into incident timelines rather than treated as independent messages. Stations will be mapped to taxi zones, and pickup and drop-off outcomes will be aggregated around the start and resolution of each event. Scheduled changes will be separated from unexpected disruptions because passengers may adapt differently when advance notice is available.

# Section 3 Data Analysis Plan

I will begin with exploratory comparisons tied to operational questions. I will compare demand by hour, weekday, season, borough, and zone activity; quantify zero and extreme demand; compare event with non-event hours around the same venues; and examine one-week and two-week lags. Weather and missingness will be compared across periods. For subway incidents, I will compare pickups and drop-offs before and after onset in affected, neighboring, and comparison zones.

## Forecasting and evaluation

The benchmark will use demand from the same zone and hour one week earlier. I will compare it with a global XGBoost model using production-available lags, calendar features, archived weather forecasts, zone identifiers, and event windows. Model selection will use validation data only; the final model will be evaluated once on a later untouched test period using MAE, RMSE, WAPE, bias, and horizon results. Performance will also be examined by borough, hour, zone activity, weather availability, events, and extreme demand so a strong citywide average cannot hide operational weaknesses.

After fixing the point forecast, I will complete three extensions. First, I will test residuals for repeatable hierarchical structure by separating between-zone variance from within-zone variation and checking stability across hours, day type, horizon, borough, and later periods. Partial pooling will be used only if the correction generalizes. Second, quantile or conformal forecasts will produce prediction intervals, evaluated through coverage and width overall and by operating segment. Third, I will deploy a dashboard showing zone forecasts, intervals, explanations, data freshness, and archived-weather lineage.

## Subway disruption causal analysis

The disruption study will ask whether ride-hailing activity changed because of a subway disruption, not merely whether both occurred together. I will construct event-time panels around unexpected incidents and compare affected station zones with similar zones not disrupted at that time. A difference-in-differences or event-study design will estimate pickup changes near affected origins and drop-off changes near likely destinations. Pre-event coefficients will test whether the groups were already moving differently. Models will account for time, weather, holidays, events, zone differences, and citywide shocks, with uncertainty clustered by incident and zone.

## Risks and decision rules

| **Risk**                      | **Why it matters**                                                                                  | **Planned response**                                                                                |
|-------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Information leakage           | Observed future weather, corrected event schedules, or incomplete trip totals could enter training. | Use forecast-origin availability checks, archived source versions, and purged chronological splits. |
| Unstable residual correction  | Visible hourly or geographic bias may not repeat over time.                                         | Estimate between- and within-zone variance and require improvement in a later validation period.    |
| Sparse or imprecise incidents | Rare events and route-level alerts may yield unstable effects or uncertain affected locations.      | Report counts and intervals, validate parsed samples, and compare narrow and broad definitions.     |
| Causal confounding            | Weather, events, maintenance, or citywide shocks may affect both transit service and ride demand.   | Use comparison zones, fixed effects, pre-trend diagnostics, and placebo tests.                      |
| Scope and computation         | Millions of rows and several extensions may exceed the capstone schedule.                           | Keep the forecasting system as the core deliverable and stage the causal study and dashboard.       |

Success will require better later-period accuracy than the weekly benchmark, valid uncertainty intervals, and direction-specific disruption effects that survive pre-trend and sensitivity checks.

# Appendix A Data Structure and Variable Description

I plan to use two connected analytical tables. The forecasting table will support one- through six-hour demand predictions, while the causal panel will measure how pickups and drop-offs change around subway disruptions. Keeping these tables separate will make the different units of analysis clear and will prevent a predictive relationship from being interpreted as a causal effect.

| **Analytical table**    | **One row represents**                | **Planned scale and content**                                                                           | **Time coverage**                                                                     |
|-------------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Forecasting table       | Forecast origin × horizon × taxi zone | About 26.1 million rows, 263 zones, six horizons, and 53 stored columns.                                | 2024-08 through 2026-06 for archived-weather modeling; demand history begins in 2021. |
| Hourly demand history   | Local hour × taxi zone                | Complete zone-hour grid with zero-demand hours retained.                                                | 2021-01 through 2026-06.                                                              |
| Subway alert timeline   | Alert update within incident          | Raw and derived incident updates, including disruption status and resolution timing.                    | Available MTA alert history; January 2021 will be the first validated period.         |
| Causal event-time panel | Incident × relative hour × zone role  | Affected, directional destination, neighboring, and comparison zones with pickup and drop-off outcomes. | Windows before, during, and after each eligible disruption.                           |

## Key variables

| **Variable group**   | **Variables**                                                                                                                          | **Description**                                                                                                                            |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Keys and timing      | `forecast_origin_utc`, `target_hour_utc`, `target_hour_local`, `forecast_horizon`, `location_id`                             | Identify when the forecast is issued, which future hour is predicted, the 1-6 hour horizon, and the TLC taxi zone.                         |
| Forecast target      | `total_pickups`                                                                                                                      | Number of HVFHV pickups in the target zone and hour.                                                                                       |
| Demand history       | `pickup_lag_24`, `pickup_lag_168`, `pickup_lag_336`                                                                              | Demand in the same zone 24 hours, one week, and two weeks earlier.                                                                         |
| Calendar             | `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_weekday_rush_hour`, `is_weekend_night`                            | Represent regular hourly, weekly, monthly, commute, and nightlife patterns.                                                                |
| Holiday and events   | `is_holiday`, `is_unofficial_holiday`, `is_post_game_demand`, `is_concert_window`, `is_wide_concert_window`                  | Mark holidays and scheduled periods when sports or concerts may change demand.                                                             |
| Archived weather     | `forecast_temperature_f`, `forecast_precipitation_in`, `forecast_rain_in`, `forecast_snowfall_in`, `forecast_wind_speed_mph` | Weather values from the forecast run available when the demand prediction would be issued.                                                 |
| Weather availability | Five variable-specific availability flags, `is_fallback_weather_run`, `selected_run_age_hours`                                     | Show whether each weather field was available and how old the selected forecast run was.                                                   |
| Geography            | `zone_latitude`, `zone_longitude`, borough and taxi-zone geometry                                                                  | Locate each zone and support mapping, borough comparisons, station matching, and future spatial features.                                  |
| Alert timeline       | `event_id`, `alert_id`, `update_number`, `date`, `status_label`, `affected`, `header`, `description`                   | Connect repeated MTA messages to the same incident and describe the affected service.                                                      |
| Disruption timing    | `next_update_time`, `minutes_until_next_update`, `is_disruption_candidate`, `mentions_resumed`                                 | Separate routine notices, measure each update interval, and identify likely disruption resolution.                                         |
| Causal outcomes      | `total_pickups`, `total_dropoffs`, relative event hour, affected-zone role, and comparison-zone flag                               | Measure substitution near outage origins and whether more passengers use ride-hailing to reach destinations inside disrupted subway areas. |

The 53-column forecasting table will contain 30 approved model features, three keys, one target, one daylight-saving sensitivity flag, and 18 metadata fields used for audits and data lineage. Metadata will remain available for validation, but it will not enter the model automatically. For the causal panel, pickup and drop-off outcomes will be analyzed separately because an increase near the disrupted origin and an increase inside the outage destination area describe different passenger responses.

# References

Miao, F., et al. (2016). Taxi dispatch with real-time sensing data in metropolitan areas: A receding horizon control approach. [<u>https://arxiv.org/abs/1603.04418</u>](https://arxiv.org/abs/1603.04418)

Rodrigues, F., Markou, I., & Pereira, F. C. (2019). Combining time-series and textual data for taxi demand prediction in event areas: A deep learning approach. Information Fusion, 49, 120-129. [<u>https://arxiv.org/abs/1808.05535</u>](https://arxiv.org/abs/1808.05535)

Safikhani, A., Kamga, C., Mudigonda, S., Faghih, S. S., & Moghimi, B. (2020). Spatio-temporal modeling of yellow taxi demands in New York City using generalized STAR models. International Journal of Forecasting, 36(3), 1138-1148. [<u>https://arxiv.org/abs/1711.10090</u>](https://arxiv.org/abs/1711.10090)

Wang, J., Zhu, W., Sun, Y., & Tian, C. (2020). An effective dynamic spatio-temporal framework with multi-source information for traffic prediction. [<u>https://arxiv.org/abs/2005.05128</u>](https://arxiv.org/abs/2005.05128)

Yao, H., et al. (2018). Deep multi-view spatial-temporal network for taxi demand prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). [<u>https://arxiv.org/abs/1802.08714</u>](https://arxiv.org/abs/1802.08714)

Zhang, J., Zheng, Y., & Qi, D. (2017). Deep spatio-temporal residual networks for citywide crowd flows prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 31(1). [<u>https://arxiv.org/abs/1610.00081</u>](https://arxiv.org/abs/1610.00081)

RideCastAI. Public fare and ETA estimation project. [<u>https://github.com/rajesh1804/RideCastAI</u>](https://github.com/rajesh1804/RideCastAI)

New York City Taxi and Limousine Commission. Trip record data. [<u>https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page</u>](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

Metropolitan Transportation Authority. Developer resources and service data. [<u>https://www.mta.info/developers</u>](https://www.mta.info/developers)

Open-Meteo. Historical weather forecast API documentation. [<u>https://open-meteo.com/en/docs/historical-forecast-api</u>](https://open-meteo.com/en/docs/historical-forecast-api)
