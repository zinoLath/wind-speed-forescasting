# Detailed study of the imputed-data forecast tests

## 1. Purpose

This document gives a detailed record of the imputed-data forecast tests.

The tests compare two bidirectional sequence models.

The first model is a bidirectional Long Short-Term Memory model.

This document uses the short name Bi-LSTM for this model.

The second model is a bidirectional Temporal Convolutional Network.

This document uses the short name TCN-Bi for this model.

The main test value is Mean Absolute Error.

This document uses the short name MAE for this value.

The final MAE is the mean of the 36 horizon MAE values.

The tests use wind speed at 100 meters as the target.

The target column name is `ws100`.

## 2. Technical terms

### 2.1 Walk-forward evaluation

Walk-forward evaluation is a time-series test method.

The method trains a model with past data.

The method then tests the model with later data.

The test time does not occur before the training time.

This order gives a test that is similar to normal operation.

### 2.2 Forecast origin

A forecast origin is the last input time before a forecast starts.

Each origin has one input sequence and one output sequence.

### 2.3 Forecast horizon

A forecast horizon is the distance from the origin to a target time.

Horizon 1 is ten minutes after the origin.

Horizon 36 is six hours after the origin.

### 2.4 Imputed value

An imputed value is a calculated replacement for a missing value.

An imputed value is not a measured value.

### 2.5 Teacher forcing

Teacher forcing gives the true prior target to the decoder during training.

This input can make the training task easier.

The true prior target is not available during a real forecast.

### 2.6 Direct decoder

A direct decoder does not use future true targets.

The tested direct decoder has two input features.

The first feature is the last available target.

The second feature is the normalized forecast horizon.

For horizon `h`, the second feature is `h / 36`.

### 2.7 Residual target

A residual target is the change from a reference value.

The reference value in these tests is the last available target.

The model predicts the residual for each horizon.

The code then adds the reference value to each prediction.

### 2.8 Persistence forecast

A persistence forecast repeats the last available target.

The same value is used for all 36 horizons.

The last available target can be an imputed value.

### 2.9 Inference

Inference is the use of a trained model to calculate a forecast.

## 3. Initial problem

The first `wfo_imputed` results had poor generalization.

Many test windows had a negative coefficient of determination.

This document uses the short name R2 for that coefficient.

The first Bi-LSTM result had an overall R2 of approximately `0.02`.

Most individual windows had a negative R2.

The initial code measured only horizon 36.

The user required good mean performance for all 36 horizons.

The test method therefore required changes before model changes could be compared.

## 4. Data audit

The imputed file contains `47,900` rows.

Each row has a ten-minute time step.

The file contains `8,868` rows with the row-level imputation flag.

This quantity is `18.51` percent of all rows.

The longest continuous imputed interval has `2,176` rows.

This interval is approximately `15.11` days.

There are `18` imputed intervals with at least 36 rows.

There are `7` imputed intervals with at least 144 rows.

The file contains `6,454` duplicate feature rows.

One August test interval has a constant imputed `ws100` value.

The value is `7.3374` for all `1,008` rows in that interval.

This interval does not contain measured target values.

It cannot give a valid measure of forecast skill.

The first result files used imputed targets as true targets.

This method mixed forecast error with imputation error.

The new test method uses the original file to identify missing targets.

The file is `data/wind_data.csv`.

The new primary metric excludes each target that was originally missing.

The output files still keep the imputation flags for additional study.

## 5. Problems in the first test method

### 5.1 One-horizon objective

The first test kept only the last output from each 36-step forecast.

Thus, the test measured only horizon 36.

This objective did not match the final study objective.

The new test keeps all 36 outputs.

### 5.2 Training and inference difference

The training decoder received prior true future targets.

The inference decoder repeated the last input target.

The model did not see this inference input pattern during training.

This difference can cause poor generalization.

### 5.3 Incorrect parameter source

The first script used fixed parameters from the script.

It did not load each model's saved Optuna result.

The first Bi-LSTM configuration used 256 units in each direction.

The saved Optuna result selected 64 units in each direction.

The larger model had much more capacity.

The new script loads each model's `best_trial.json` file by default.

### 5.4 Synthetic targets in the metric

The first primary metric included imputed target values.

An imputed target is not independent test data.

The new primary metric uses only measured targets.

### 5.5 Noncausal wavelet operation

The wavelet operation used the full test section.

Thus, a value near the forecast origin could use later test data.

This operation is noncausal.

The formal benchmark uses raw `ws100` without wavelet denoising.

No wavelet against raw performance test was done.

The raw target was selected to remove this source of future data.

### 5.6 Invalid legacy rolling window

The old rolling method moved the encoder by one origin.

It then added the target from 36 steps after that origin.

This action made a time gap inside the next encoder window.

The corrected method adds the row at the next input time.

### 5.7 Result file mixture

The first script read all matching files in one model directory.

Old files could remain after an incomplete run.

The new script uses a separate directory for each named run.

The new script also joins only predictions from the current process.

## 6. Formal benchmark method

### 6.1 Common settings

The formal comparison uses two test windows.

Each test window contains seven days.

The step between test windows is 30 days.

The training window contains 60 days unless a test changes this value.

The input sequence contains 72 time steps.

The input duration is 12 hours.

The output sequence contains 36 time steps.

The output duration is six hours.

The maximum epoch quantity is 20.

Early stopping patience is four epochs.

The batch size is 32.

The random seed is 42.

The benchmark uses the saved Optuna parameters.

The benchmark uses raw `ws100` as the target.

### 6.2 Training and validation split

Each training window has an 80 percent training section.

The last 20 percent is the validation section.

The split follows time order.

The scaler uses only the training section.

The validation and test sections use the training scaler.

### 6.3 Test context

The test input includes the last 72 rows of the training window.

This context permits a forecast at the start of the test window.

The test does not discard the first 12 test hours.

### 6.4 Prediction rows

A seven-day test window has `1,008` time steps.

The 36-step output permits `973` complete forecast origins.

Each origin has 36 prediction rows.

One complete test window has `35,028` origin-horizon rows.

Two complete test windows have `70,056` rows before target removal.

The formal benchmark has `69,696` measured-target rows.

The metric excludes `360` rows with imputed targets.

### 6.5 Equal horizon weight

The script calculates one MAE value for each horizon.

The script then calculates the arithmetic mean of these 36 values.

Thus, each horizon has the same weight.

The script also keeps the pooled MAE for reference.

The persistence MAE uses the same horizon mean method.

### 6.6 MAE formula

For one horizon, the absolute error is:

```text
absolute_error = abs(actual - predicted)
```

The horizon MAE is:

```text
horizon_MAE = sum(absolute_error) / measured_target_count
```

The final MAE is:

```text
final_MAE = sum(horizon_MAE_1 ... horizon_MAE_36) / 36
```

### 6.7 Comparison rule

Only one main change was made for each test.

The same time windows were used when this was possible.

A change was kept when it decreased the final MAE.

A change was rejected when it increased the final MAE.

Small results from one window were not sufficient for a strong decision.

## 7. Baseline results

The original all-horizon Bi-LSTM MAE was `1.5648`.

The original all-horizon TCN-Bi MAE was `1.5632`.

The persistence MAE was `1.5921`.

Both models were only a small amount better than persistence.

The original validation loss was much lower than the inference error.

This difference gave evidence of a decoder input problem.

## 8. Test 1: direct decoder

### 8.1 Change

The direct decoder used no future true target.

The first decoder feature repeated the last available target.

The second decoder feature identified the forecast horizon.

### 8.2 Expected result

The training input and inference input became equal.

This equality was expected to decrease the decoder input shift.

### 8.3 Bi-LSTM result

The Bi-LSTM MAE decreased from `1.5648` to `1.4829`.

This decrease was approximately `5.2` percent.

The direct decoder was kept for Bi-LSTM.

### 8.4 TCN-Bi result

The TCN-Bi MAE increased from `1.5632` to `1.5997`.

This increase was approximately `2.3` percent.

The direct decoder was rejected for TCN-Bi.

### 8.5 Study note

One decoder method did not give the best result for both models.

The recurrent decoder used the horizon feature well.

The tested TCN decoder did not get the same benefit.

The TCN decoder has a short causal receptive field.

Its direct input can become too similar at later horizons.

## 9. Test 2: residual target

### 9.1 Change

The model predicted a change from persistence.

The target in scaled units was:

```text
residual_target = future_scaled_target - last_scaled_target
```

The final prediction was:

```text
predicted_scaled_target = predicted_residual + last_scaled_target
```

### 9.2 Bi-LSTM result

The Bi-LSTM MAE decreased from `1.4829` to `1.4537`.

This decrease was approximately `2.0` percent.

The residual target was kept for Bi-LSTM.

### 9.3 TCN-Bi result

The first TCN-Bi residual test had an MAE of `1.6436`.

The corrected validation test had an MAE of `1.9194`.

Both values were worse than the absolute-target result.

The residual target was rejected for TCN-Bi.

### 9.4 Study note

Residual prediction gives the model a persistence reference.

This reference can make the target range smaller.

This method helped the recurrent model.

It did not help the tested convolutional model.

## 10. Test 3: inference-style validation

### 10.1 Problem

TCN-Bi training used teacher forcing.

The first validation also used teacher forcing.

Inference did not have true future decoder targets.

Thus, early stopping selected an epoch for a different input condition.

### 10.2 Change

Training continued to use teacher forcing.

Validation repeated the origin target for all decoder steps.

Inference used the same decoder input pattern.

### 10.3 Result

The TCN-Bi MAE decreased from `1.5632` to `1.4721`.

This decrease was approximately `5.8` percent.

The TCN-Bi R2 increased from approximately `0.129` to `0.257`.

The inference-style validation was kept.

### 10.4 Study note

Early stopping is part of the model selection process.

A good training loss does not prove good inference performance.

Validation must use conditions that are close to operation.

The training and validation decoder inputs are still different for TCN-Bi.

This difference is a remaining limit.

## 11. Test 4: circular direction features

### 11.1 Change

Each direction value was changed to sine and cosine features.

This change removes the numerical break between 359 degrees and 1 degree.

### 11.2 Result

The Bi-LSTM MAE increased from `1.4829` to `1.5245`.

The TCN-Bi MAE increased from `1.5632` to `1.5747`.

The change was rejected for both models.

### 11.3 Possible causes

The change increased the input feature quantity.

The saved Optuna parameters were selected for the old feature set.

The new feature set can require new parameter selection.

The wind speed columns can already contain much of the useful information.

### 11.4 Study note

The physical form of a feature can be correct without a test improvement.

A feature change can require a new model search.

## 12. Test 5: removal of imputed training targets

### 12.1 Change

The test removed each training sequence with an imputed target in its output.

The primary evaluation metric already removed imputed test targets.

### 12.2 Result

The Bi-LSTM MAE increased from `1.4829` to `1.6244`.

This increase was approximately `9.5` percent.

The TCN-Bi MAE increased from `1.5632` to `1.5834`.

This increase was approximately `1.3` percent.

The change was rejected for both models.

### 12.3 Possible causes

The removal decreased the training sample quantity.

It also removed sequences near missing-data intervals.

These sequences can contain useful measured input values.

Continuous training coverage was more useful than strict target removal.

### 12.4 Study note

Imputed targets are not suitable for the primary test metric.

However, they can still supply useful training continuity.

Training rules and evaluation rules do not have to be equal.

## 13. Test 6: training window length

### 13.1 Method

The test compared 30, 60, and 90 training days.

All three tests used the same February test interval.

### 13.2 Results

| Training days | Bi-LSTM MAE | TCN-Bi MAE | Model mean |
|---:|---:|---:|---:|
| 30 | 1.2917 | 1.6367 | 1.4642 |
| 60 | 1.2995 | 1.5985 | 1.4490 |
| 90 | 1.3617 | 1.5919 | 1.4768 |

The 30-day window gave the best Bi-LSTM result for this interval.

The 90-day window gave the best TCN-Bi result for this interval.

The 60-day window gave the best mean result for both models.

The 60-day window was kept as the common default.

### 13.3 Study note

A short window gives more weight to recent weather conditions.

A long window gives more samples and more weather conditions.

Old data can have a different distribution from current data.

The best window can be different for each model.

One test interval is not sufficient for a final model-specific window rule.

## 14. Test 7: MAE training loss

### 14.1 Change

The model used MAE as its training loss instead of Mean Squared Error.

This document uses the short name MSE for Mean Squared Error.

### 14.2 Bi-LSTM February result

The February MAE decreased from `1.2995` to `1.2527`.

This result first appeared to support MAE loss.

### 14.3 Bi-LSTM January result

The January MAE increased from `1.6078` to `1.7791`.

The change did not generalize to the second weather interval.

The two-window MAE increased from `1.4537` to `1.5159`.

This increase was approximately `4.3` percent.

MAE loss was rejected for Bi-LSTM.

### 14.4 TCN-Bi result

The February TCN-Bi MAE increased from `1.5985` to `1.6114`.

MAE loss was rejected for TCN-Bi.

### 14.5 Study note

One time interval can give a false positive result.

Tests must include different weather conditions.

An evaluation MAE objective does not require an MAE training loss.

MSE gives more weight to large errors.

This weight can help model stability.

## 15. Test 8: larger TCN receptive field

### 15.1 Change

The first TCN encoder used dilation values `[1, 2]`.

The test used dilation values `[1, 2, 4, 8]`.

The larger set lets the TCN use a longer input distance.

### 15.2 Result

The February baseline MAE was `1.5985`.

The larger receptive-field MAE was `1.6223`.

The change increased training time and error.

The change was rejected.

### 15.3 Possible causes

The saved filters and learning rate were selected for the short dilation set.

The larger network can require a different learning rate.

The added distance can also add data that is not useful.

### 15.4 Study note

A receptive field must be large enough for the required signal.

A larger receptive field does not always give a better result.

## 16. Parameter-source correction

This correction was required for a fair test.

It was not a separate model performance test.

The script now loads the parameters for each wrapper name.

The Bi-LSTM saved result uses 64 units in each direction.

The learning rate is approximately `0.006403`.

The encoder dropout value is `0.05`.

The decoder dropout value is `0.15`.

The default fallback values now agree with this saved result.

## 17. Final retained configurations

### 17.1 Bi-LSTM

The Bi-LSTM uses a direct decoder.

The decoder receives the last available target.

The decoder also receives the normalized horizon.

The Bi-LSTM predicts a residual from persistence.

The Bi-LSTM uses MSE loss.

The Bi-LSTM uses its saved Optuna parameters.

The final two-window MAE is `1.4537`.

The improvement from the original Bi-LSTM is approximately `7.1` percent.

The improvement from persistence is approximately `8.7` percent.

### 17.2 TCN-Bi

The TCN-Bi uses teacher forcing during training.

The TCN-Bi uses inference-style decoder inputs during validation.

The TCN-Bi predicts the absolute target.

The TCN-Bi uses MSE loss.

The TCN-Bi uses its saved Optuna parameters.

The final two-window MAE is `1.4721`.

The improvement from the first TCN-Bi baseline is approximately `5.8` percent.

The improvement from persistence is approximately `7.5` percent.

## 18. Selected horizon results

The following table shows selected horizons from the retained configurations.

| Model | Horizon | Model MAE | Persistence MAE | R2 |
|---|---:|---:|---:|---:|
| Bi-LSTM | 1 | 0.6638 | 0.5848 | 0.8255 |
| Bi-LSTM | 6 | 1.0982 | 1.1371 | 0.5663 |
| Bi-LSTM | 12 | 1.3191 | 1.4266 | 0.4110 |
| Bi-LSTM | 18 | 1.4905 | 1.6600 | 0.2813 |
| Bi-LSTM | 24 | 1.6388 | 1.8465 | 0.1336 |
| Bi-LSTM | 30 | 1.7789 | 2.0006 | -0.0169 |
| Bi-LSTM | 36 | 1.8801 | 2.0753 | -0.1250 |
| TCN-Bi | 1 | 0.9783 | 0.5848 | 0.6879 |
| TCN-Bi | 6 | 1.1741 | 1.1371 | 0.5165 |
| TCN-Bi | 12 | 1.3573 | 1.4266 | 0.3573 |
| TCN-Bi | 18 | 1.4998 | 1.6600 | 0.2465 |
| TCN-Bi | 24 | 1.6317 | 1.8465 | 0.1253 |
| TCN-Bi | 30 | 1.7292 | 2.0006 | 0.0344 |
| TCN-Bi | 36 | 1.7864 | 2.0753 | -0.0154 |

Persistence is better than both models at horizon 1.

The models become better than persistence at many later horizons.

The Bi-LSTM is better than TCN-Bi at many early and middle horizons.

The TCN-Bi is better at the selected long horizons.

R2 decreases as the horizon increases.

This decrease shows that long-range prediction is still difficult.

## 19. Changes to the test script

The script now stores one row for each origin and horizon.

Each prediction row contains these fields:

- forecast origin;
- target time;
- horizon number;
- actual target;
- model prediction;
- persistence prediction;
- input imputation flag;
- output target imputation flag.

The script calculates one metric set for each horizon.

The script calculates the mean of the horizon MAE values.

The script keeps pooled metrics for additional analysis.

The script rejects overlapping test windows.

This rule prevents the same forecast from receiving more than one weight.

The script uses a separate result directory for each run name.

The script records the resolved model parameters in `metrics.json`.

The script records the decoder mode and target mode.

The script sets the TensorFlow random seed for each window.

The script requests deterministic TensorFlow operations.

## 20. Changes to the sequence wrapper

The sequence wrapper supports `teacher_forcing` and `direct` decoder modes.

The sequence wrapper supports `absolute` and `residual` target modes.

The decoder feature quantity now comes from the prepared decoder data.

This change permits the two-feature direct decoder.

The prediction method reconstructs residual predictions before inverse scaling.

The rolling method uses the correct next input row.

Existing scripts keep teacher forcing and absolute targets by default.

Only benchmarked bidirectional models receive automatic retained modes in `wfo_imputed`.

## 21. Reproduction

Use the project Python 3.10 environment.

Load the GPU environment before a TensorFlow run.

Run the retained benchmark with this command:

```bash
source .venv/bin/activate
source scripts/tf_gpu_env.sh
python tests/wfo_imputed.py \
  --models lstm_bi tcn_bi \
  --train-window-days 60 \
  --test-window-days 7 \
  --step-days 30 \
  --epochs 20 \
  --patience 4 \
  --max-windows 2 \
  --seed 42 \
  --run-name retained_all_horizon_study
```

The automatic mode selects the retained mode for each bidirectional model.

Bi-LSTM receives direct decoding and residual targets.

TCN-Bi receives teacher forcing and absolute targets.

Both models receive MSE loss.

## 22. Result files

Each run writes files below `data/wfo_results_imputed`.

Each model and run name has a separate directory.

`metrics.json` contains settings, window metrics, and overall metrics.

`metrics.csv` contains one metric row for each test window.

`predictions_all.csv` contains all origin-horizon predictions.

Each `predictions_window_XXXX.csv` file contains one test window.

The benchmark run files used for this study remain in the data directory.

## 23. Limits of this study

The formal comparison uses only two main test windows.

Two windows do not represent all seasons and all weather conditions.

Some secondary tests use only the February window.

These tests give weaker evidence than the two-window tests.

The imputer was fitted with the full imputed-data process.

This process can use information from later times.

Thus, input imputation is not fully causal.

The primary metric removes imputed targets but not imputed inputs.

The persistence reference can be an imputed value.

TCN-Bi still uses different decoder inputs for training and inference.

Inference-style validation reduces this problem but does not remove it.

The Optuna parameters came from an earlier objective.

They were not selected again for the final all-horizon objective.

The feature set has many related wind-height columns.

The study did not do a complete feature selection test.

The study did not calculate prediction intervals.

The study did not test forecast uncertainty.

## 24. Recommended next studies

### 24.1 More walk-forward windows

Use all valid non-overlapping windows.

Report the mean, median, and standard deviation of window MAE.

Report results for each season.

### 24.2 Causal imputation

Fit the imputer with training data from each walk-forward window.

Do not use a later value to fill an earlier input.

Compare causal imputation with the current imputed file.

### 24.3 New Optuna search

Use the mean 36-horizon MAE as the Optuna objective.

Use inference-style validation during each trial.

Search each retained model separately.

### 24.4 Horizon groups

Report short, middle, and long horizon groups.

For example, use horizons 1 through 6 as the short group.

Use horizons 7 through 18 as the middle group.

Use horizons 19 through 36 as the long group.

### 24.5 Persistence combination

Persistence is strong at horizon 1.

Test a horizon-dependent combination of persistence and model predictions.

Use validation data to select the combination weights.

### 24.6 Imputation-mask features

Add a feature that identifies each imputed input value.

Do not use only one row-level flag.

Test whether the model learns to reduce weight for synthetic inputs.

### 24.7 Feature study

Test a smaller set of wind-height features.

Test vertical wind shear and vertical gradient features.

Repeat Optuna selection after each large feature change.

## 25. Main conclusions

The first poor results did not have one cause.

The metric, decoder inputs, parameter source, and target source all affected the result.

Bi-LSTM required direct decoding and residual targets.

TCN-Bi required inference-style validation.

Changes that were physically reasonable did not always decrease MAE.

Circular features and a larger receptive field are examples.

One-window improvements did not always generalize.

The MAE-loss test is the clearest example.

Measured-target metrics are necessary for an imputed-data study.

Equal horizon weight is necessary for the selected 36-horizon objective.

The retained models now have a clear and reproducible test method.
