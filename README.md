# Can process sensors warn us before a unit fails final test?

ES 335 (Machine Learning) course project, predicting final-test failures on the SECOM semiconductor manufacturing dataset.

- **Live website:** https://sachethpraveen-afk.github.io/ES335_Machine_Learning_Project_24110308_24110299/
- **Source repository:** https://github.com/sachethpraveen-afk/ES335_Machine_Learning_Project_24110308_24110299
- **Team:** Risvan K (24110299) and Sacheth C Praveen (24110308)

## The question

Using only the 590 process-sensor readings recorded for a manufactured unit, can we rank units so that those that later fail final test appear near the top of the list? A row is one unit. The target is 1 if the unit failed final test and 0 if it passed. Only the sensor values are used as inputs. The claim is limited to ranking units from one process in one three-month window (see Limitations).

## Repository contents

| File | What it is |
|---|---|
| `index.html` | The interactive website (self-contained: the test-set scores are embedded, no server needed). Served by GitHub Pages. |
| `ML_project_24110299_24110308.ipynb` | The full analysis: data loading, quality checks, split, preprocessing, baseline, model comparison, final test evaluation, error analysis. **All code is in this notebook.** |
| `test_predictions.csv` | Test-set scores from both final models (written by the notebook). The website's interactive chart uses these scores. |
| `requirements.txt` | Python packages needed to rerun the notebook. |
| `data/` | Place the two SECOM files here (see Data access). |

## Data access

The data are the public **SECOM** dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/179/secom

- 1,567 units, 590 sensor features, 104 failures (6.64%), recorded 2008-07-19 to 2008-10-17.
- Two files are needed: `secom.data` (sensor values) and `secom_labels.data` (label -1 = pass, 1 = fail, plus a timestamp).
- Download the dataset from the UCI link above, unzip it, and place `secom.data` and `secom_labels.data` in the `data/` folder (or upload them to `/content/` if you use Google Colab).
- The raw data files are not modified in any way; all cleaning happens inside the notebook.

## Setup and how to rebuild the results

1. Clone the repository and install the packages:
   ```bash
   git clone https://github.com/sachethpraveen-afk/ES335_Machine_Learning_Project_24110308_24110299
   cd YOUR-REPO
   pip install -r requirements.txt
   ```
2. Download the two SECOM files as described above.
3. Open `ML_project_24110299_24110308.ipynb` in Jupyter or Google Colab.
4. The first cell reads the files from `/content/secom.data` and `/content/secom_labels.data` (the Colab layout). If you run locally, change those two paths to `data/secom.data` and `data/secom_labels.data`.
5. Run all cells from top to bottom. The last evaluation cell writes `test_predictions.csv`.

Random seed 42 is used for the random forest and the bootstrap, so results should reproduce.

## Method summary

- **Split (time-ordered, no shuffling):** first 60% of units for training, next 20% for validation, last 20% (314 units, 17 failures) held out as the test set. Tuning used time-series cross-validation (5 folds) on the first 80% only.
- **Preprocessing (fit on training data only, inside a pipeline to avoid leakage):** median imputation, removal of constant features, standardisation, and univariate feature selection (`SelectKBest`, ANOVA F-test).
- **Baseline:** a no-skill classifier that predicts the failure rate, so PR-AUC equals the failure rate of the evaluated block.
- **Models:** logistic regression (`class_weight="balanced"`, C = 0.01, k = 10 features) as the primary model, and a random forest (k = 100 features, 300 trees) as a secondary comparison.
- **Metrics:** PR-AUC and its gain over the no-skill rate (primary), recall at precision >= 0.15, and recall within the top 10% of flagged units. Accuracy is not used because only about 6% of units fail.
- **Uncertainty:** bootstrap resampling of whole calendar days (2,000 resamples), because failures occur in bursts.
- **Test discipline:** model settings were fixed using development data only; the test set was evaluated once.

## Results (test set, evaluated once)

| Model | PR-AUC | Gain over no-skill (95% CI) | Recall at precision >= 0.15 | Recall in top 10% |
|---|---|---|---|---|
| No-skill baseline | 0.054 | 0 | n/a | n/a |
| Logistic regression (k = 10) | 0.227 | median 0.195 [0.017, 0.450] | 0.53 | 0.35 |
| Random forest (k = 100) | 0.107 | median 0.061 [0.018, 0.121] | 0.00 | 0.18 |

The logistic regression ranked failures better than chance, but the confidence interval is wide and the forest-minus-logistic difference (median -0.134, 95% CI [-0.376, 0.031]) includes zero, so we do not claim the simpler model is definitively better.

## Limitations

- Only 17 failures in the test set and 16 test days, so estimates are uncertain.
- Failures cluster in time (many occur on 2008-10-05 and 2008-10-06), so the units are not independent.
- One process, one three-month window, one time-ordered split: results may not transfer to other periods or fabs.
- The two models differ in both algorithm and number of selected features, so the comparison mixes model and setting.
- Other model families (such as gradient boosting) were not tested.

## AI-use disclosure

We used **Claude (Anthropic)** for project planning, explanations of methods, code drafts and drafting of the website text. We ran the code on the data ourselves, checked the outputs, and made the modelling and evaluation decisions listed above. Any text or code we did not fully understand was not kept. Each team member can explain their contribution and the choices made.

## Citations and external material

- SECOM [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/179/secom
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Libraries used: pandas, NumPy, scikit-learn, Matplotlib.

## Security and data handling

The repository contains no credentials, API keys or private data. The only data used are the public SECOM files.
