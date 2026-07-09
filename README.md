# MLOps Capstone 2 — Casting Defect Detection

Skeleton for the **Casting Defect Detection** Deep-Learning MLOps pipeline (ResNet18 transfer
learning). Infrastructure + the test harness are provided; **you implement the modelling + MLOps
logic** marked `# TODO`. Each notebook section's marks (red) match `MLOps2_Grading_Rubric.xlsx`.

## Provided (don't rewrite)
`config.py` · `src/__init__.py` · `src/dataset.py` · `requirements.txt` · `Dockerfile` ·
`.github/workflows/ci.yml` · `tests/`

## You build (the `# TODO`s)
| File | Stage | You implement |
|---|---|---|
| `Data_Preparation.ipynb` | 1, 2.1 | guided notebook: understanding, quality, versioning, EDA, preprocessing |
| `Model_Development_and_Tracking.ipynb` | 2.2–2.4, 3 | guided notebook: transfer learning, MLflow, eval, registry, FastAPI/Docker/CI |
| `Operations_Monitoring_and_Evidence.ipynb` | 4 | guided notebook: logging, statistical + embedding drift, retraining, rollback |
| `src/data_prep.py` | 1, 2.1 | discovery, quality validation, versioned splits, transforms, image features |
| `src/model.py` | 2.2, 4.3 | ResNet18 transfer-learning model + embedding extractor |
| `src/train.py` | 2.3–2.4, 3.2 | training loop, MLflow tracking + registry + @production, reference baseline |
| `src/evaluate.py` | 3.1 | recall-on-defect / F1 / ROC / confusion + failure cases |
| `app.py` | 3.3 | FastAPI `/health` + `/predict` (image upload) + prediction logging |
| `src/monitoring.py` | 4.1–4.3 | statistical (Evidently+PSI) + embedding drift + confidence |
| `src/retrain.py` | 4.4–4.5 | drift-triggered retrain, version compare, promote/rollback |

The three notebooks carry **descriptive per-task instructions** (objective · stage/task · inputs→outputs
· TODO expectations · dependencies · where to document). In addition, each stage now opens with a
**Markdown sub-task checklist** — every sub-task shows its ID and marks (e.g. `2.2.1 — ImageNet-pretrained
ResNet18, backbone frozen [4]`) so you can see exactly what each mark rewards. Stage 4.3 in the operations notebook also includes a **conceptual walkthrough of embedding drift** (penultimate-layer 512-dim embedding → distance-to-centroid → PSI) so you are not left
to infer it.

## Setup
```bash
pip install -r requirements.txt
# Download the casting dataset (Kaggle) and extract into data/ so you have:
#   data/.../train/{ok_front,def_front}/   and   data/.../test/{ok_front,def_front}/
#   https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
```

## Run (after implementing the TODOs)
```bash
python -m src.train          # train → MLflow → register @production
python -m src.monitoring     # statistical + embedding drift → drift_summary.json
python -m src.retrain        # drift-triggered retrain + rollback decision
pytest -q                    # API + pipeline tests
uvicorn app:app --port 8000  # serve; POST an image to /predict
```

## What you submit 
Upload these individual files; all 100 marks are graded from them:
1. `Casting_Defect_MLOps_Report` as **PDF** (Stages 1, 3, 4 + Stage-3/4 evidence).
2. `Data_Preparation.ipynb` (executed, with outputs).
3. `Model_Development_and_Tracking.ipynb` (executed, with outputs).
4. `Operations_Monitoring_and_Evidence.ipynb` (executed, with outputs).
5. `app.py`.

**Note:** Repository-generated evidence (MLflow UI, Docker build, CI green run, drift report) must be captured
**inside the notebooks/report** as screenshots, code output and summaries — we don't accept
repositories, ZIPs, HTML or JSON files.

## Acceptance targets
* `pytest` green; `/health` returns `model_loaded: true` after training.
* Test **recall on defects** is the headline metric; report F1 + ROC-AUC + confusion.
* `monitoring` flags **statistical and embedding** drift; `retrain` makes a promote/rollback decision.
* MLflow registry shows `casting_defect_classifier` with a `@production` alias.

  MLFlow UI commands:
  * mlflow ui --backend-store-uri file:D:/py_test/gdrive_sync/upGrad-learning/Python/casting-defect-mlops/mlruns
  * http://127.0.0.1:5000 (in browser)
<h2>Module 1 - S1 - Capstone Workflow and Resources</h2>


Welcome to the MLOps Capstone!

In this capstone, you will apply the concepts and techniques learned throughout the programme to design and build an end-to-end machine learning operations (MLOps) system for a real-world problem. The capstone will require you to analyse data, make design decisions, develop machine learning solutions, implement production workflows, monitor model behaviour, and communicate your findings effectively.

To help you get started, we have provided the Capstone Starter Package below, which contains:
- **Capstone Workflow Document:** Your primary guide for the capstone
- **Capstone Dataset:** Used throughout the capstone for analysis, development, and evaluation

Before starting the implementation, read the workflow document thoroughly. It explains the business problem, capstone objectives, system architecture, project stages, deliverables, evaluation framework, and assessment criteria that will guide your work throughout the capstone.

Begin by extracting the contents of the starter package and familiarising yourself with the workflow document and dataset. Refer to the workflow document regularly throughout the capstone, as it serves as the source of truth for the expected outputs, implementation requirements, and evaluation criteria for each stage of the capstone.


<h2>Module 1 - S2 - Capstone Starter Files</h2>

<!-- #region -->
The Capstone Implementation Package below contains all the starter notebooks, supporting files, utility scripts, datasets, and resources required to build your solution. These files provide the implementation framework for each stage of the capstone and are intended to guide you through the development process.

Begin by extracting the implementation package and reviewing the folder structure carefully. Complete your work only in the designated sections of the starter notebooks and files.


Important:

- Do not rename, delete, or reorganise the provided files and folders.
- Do not modify pre-written notebook instructions, utility scripts, test files, datasets, function signatures, or evaluation logic unless explicitly instructed.
- Implement your solution only in the marked solution areas provided in the starter notebooks and files.
- Execute the notebooks in the recommended order and save your work regularly.

The provided files form the reference implementation environment for the capstone. Any modifications outside the designated solution areas may lead to execution issues and can affect the evaluation of your submission.
<!-- #endregion -->

```bash

```
