## Environment
Create the environment using environment.yaml
Create the environment defined in `environment.ya


## Datasets

Download IAM Dataset. You would have to create an IAM account. Download the offline lines dataset and ground truths.

Run `generate-all-datasets.sh` located inside data_processing/ folder

```
cd data_processing
./generate-all-datasets
```

## Stroke Recovery

An example config with a model and weights are located inside example_weights/ folder for offline data. execute the file `stroke_recovery_offline.py`. 
Ensure that the required pngs and ground truths are present after data preparation in the previous step.

```
python stroke_recovery_offline.py
```

## Stroke Transformation

After extracting the .npy file run `transform_strokes.ipynb` to create the corresponding files to be used in Approaches 1 and 2.
