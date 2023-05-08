## Environment
Create the environment using environment.yaml
Create the environment defined in `environment.ya

## External Repository
Clone the Simple HWR repositor, which is a close implementation of the paper https://arxiv.org/abs/2105.11559
```
git clone https://github.com/Tahlor/simple_hwr .
```
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


----------------------------

(This code has been built on the existing repository : https://github.com/Tahlor/simple_hwr, which is an implementation of the research paper : https://arxiv.org/abs/2105.11559)