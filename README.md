# ExpGCN

This is the official implementation of our paper under review:

> ExpGCN: Review-aware Graph Convolution Network for Explainable Recommendation

----------

## Requirements
The model implementation ensures compatibility with the Recommendation Toolbox [RecBole](https://recbole.io/) (Github: [Recbole](https://github.com/RUCAIBox/RecBole)), and use [Numba](https://numba.pydata.org/) for high speed nagative sampling.

- Python: 3.8+
- RecBole: 1.0.1
- Numba: 0.55.1+

## Data Formulation

The dataset is processed as follows:

#### <DATASET_NAME>.inter

Include all $\langle user, item, explanation \rangle$ triplet interaction data. Each row contains a $\langle user, item \rangle$ pair, where the triplet data is formulated as
```html
<user_id>   <item_id>   <explanation_id_1>,<explanation_id_2>...
```

#### <DATASET_NAME>.item
Include name of all items. Each row contains an item identified by `<item_id>`.

#### <DATASET_NAME>.user
Include name of all users. Each row contains a user identified by `<user_id>`.

The above three files are placed in folder `Data/<DATASET_NAME>`. The configuration file can be created in `Params/<DATASET_NAME>.yaml`. In this repository we have provided [Amazon Movies & TV](https://github.com/lileipisces/EXTRA) (AmazonMTV) dataset as an example.

## Run

The script `run.py` is used to run the demo. Train and avaluate ExpGCN on a specific dataset, run
```python
python run.py --dataset DATASET_NAME
```