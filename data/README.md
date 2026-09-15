# Data Files

This folder contains the OFETNow database snapshot, processed training features and targets, an input template, and example inputs for the **OFETNow-ML** prediction workflow.


## Files

### `input_template.json`

A blank JSON template for running predictions on new polymer OFET candidates.

Input fields:

```text
entry_id
polymer_name
SMILES
Mn_Kda
N_AVE
PDI
Aspect_Ratio
Capacitance
GATE
Dielectric
Fabrication_Method
Solvent_of_Spin_Coating
Test_Environment
Type
S/D
Annealing_Temperature
```

`Mn_Kda` and `N_AVE` are both included for flexibility. Users may provide either value depending on available characterization data. The preprocessing workflow handles the feature conversion required by the released models.

For categorical variables, it is strongly recommended to use the feature names and category names provided in the feature-name files under `models/`. This helps the fitted one-hot encoder correctly recognize the input categories and improves prediction reliability.

### `example_input.json`

An example JSON input file with five polymer OFET entries for testing the prediction workflow.

### `ofetnow_export_fulldata.csv`
A snapshot of the complete OFETNow database as of June 3, 2026.

### `training_features.csv`

The processed training matrix contains 512 entries and the 2,213 columns listed in `models/all_features.txt`.

### `training_targets.csv`

The training targets contain the binary mobility labels (`labels1`) and three-class carrier-type labels (`labels2`).

The model-training notebook reads the released feature-name files directly to construct the 244-feature mobility-classifier matrix and the 334-feature carrier-type-classifier matrix.
The ten random seeds used for repeated cross-validation are archived in `models/cv_random_seeds.txt`.


## Input Format

JSON is the recommended input format for new predictions because it clearly preserves the field structure for each polymer OFET entry.

CSV input is also supported. The prediction workflow can read a `.csv` file and automatically convert it to the JSON list-of-records format before running inference.

Each JSON input file should contain a list of entries:

```json
[
  {
    "entry_id": "EX001",
    "polymer_name": "example_polymer",
    "SMILES": "example_smiles",
    "Mn_Kda": null,
    "N_AVE": null,
    "PDI": null,
    "Aspect_Ratio": null,
    "Capacitance": null,
    "GATE": "",
    "Dielectric": "",
    "Fabrication_Method": "",
    "Solvent_of_Spin_Coating": "",
    "Test_Environment": "",
    "Type": "",
    "S/D": "",
    "Annealing_Temperature": null
  }
]
```

Missing numerical values should be written as `null`. Missing categorical values should be written as an empty string `""` only when the value is genuinely unavailable.


## Citation

If you use these data or the prediction workflow, please cite the associated OFETNow-ML article: [10.1021/jacs.6c11486](https://doi.org/10.1021/jacs.6c11486).
