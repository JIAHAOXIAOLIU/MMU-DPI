# MMU-DPI

Enhancing Generalization in Drug–Protein Interaction Prediction through Multimodal Learning and a Label Mix Strategy

## Data 

### Conventional Benchmark Datasets

MMU-DPI was evaluated on four conventional drug–protein interaction benchmark datasets:BindingDB、Davis、KIBA、Luo's dataset

Because of GitHub storage limitations, the processed datasets used in this study can be downloaded from the following Google Drive folder:

https://drive.google.com/drive/folders/15y08E8A7yL4TLAVch8TX0cnFcK_AeR_s?usp=drive_link

### Strict Generalization Dataset

DUD-E was used for strict unseen-node generalization evaluation.

The original DUD-E dataset is available from its official website:

https://dude.docking.org/

The processed DUD-E data used in this study can also be downloaded from the Google Drive folder provided above.

### Independent Cross-Database Evaluation

The independent cross-database evaluation used the processed DrugBank benchmark as the model-development dataset and DrugMAP 2.0 exclusively as the external test database.

#### DrugBank

The processed DrugBank benchmark was obtained from the EDeepDTI data and code release:

https://doi.org/10.5281/zenodo.13825147

The DrugBank-derived files are not redistributed directly in this repository. Users should obtain the data from the original source and comply with the applicable data-use and licensing requirements.

#### DrugMAP 2.0

The original DrugMAP 2.0 data are available from the official database:

https://drugmap.idrblab.net/full-data-download

DrugMAP was used exclusively for independent external evaluation and was not used for model training, validation, hyperparameter selection, early stopping, model selection, or fine-tuning.

The DrugMAP preprocessing procedure included feature filtering, identifier matching, pair-level de-overlapping against the complete processed DrugBank positive set, and construction of the Seen-seen, Cold-drug, Cold-target, and Cold-both evaluation settings.
