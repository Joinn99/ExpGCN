# ExpGCN: Review-aware Graph Convolution Network for explainable recommendation

This is the official PyTorch implementation of our paper:

> T. Wei, T. W. S. Chow, J. Ma, and M. Zhao, “ExpGCN: Review-aware Graph Convolution Network for explainable recommendation,” in *Neural Networks*, 2022. [Paper Link](https://doi.org/10.1016/j.neunet.2022.10.014) 

<kbd>![CARAR](https://raw.githubusercontent.com/Joinn99/RepositoryResource/master/ExpGCN_Architechture.svg)</kbd>

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

## How to cite

```bibtex
@ARTICLE{Wei2022Exp,
  author   = {Tianjun Wei and Tommy W.S. Chow and Jianghong Ma and Mingbo Zhao},
  journal  = {Neural Networks},
  title    = {ExpGCN: Review-aware Graph Convolution Network for explainable recommendation},
  year     = {2022},
  issn     = {0893-6080},
  abstract = {Existing works in recommender system have widely explored extracting reviews as explanations beyond user-item interactions, and formulated the explanation generation as a ranking task to enhance item recommendation performance. To associate explanations with users and items, graph neural networks (GNN) are usually employed to learn node representations on the heterogeneous user-item-explanation interaction graph. However, modelling heterogeneous graph convolution poses limitations in both message passing styles and computational efficiency, resulting in sub-optimal recommendation performance. To address the limitations, we propose an Explanation-aware Graph Convolution Network (ExpGCN). In particular, the heterogeneous interaction graph is divided to subgraphs regard to the edge types in ExpGCN. By aggregating information from distinct subgraphs, ExpGCN is capable of generating node representations for explanation ranking task and item recommendation task respectively. Task-oriented graph convolution can not only reduces the complexity of heterogeneous node aggregation, but also alleviates the performance degeneration caused by the conflicts between task learning objectives, which has been neglected in current studies. Extensive experiments on four public datasets show that ExpGCN significantly outperforms state-of-the-art baselines with high efficiency, demonstrating the effectiveness of ExpGCN in explainable recommendations.},
  doi      = {https://doi.org/10.1016/j.neunet.2022.10.014},
  groups   = {Aspect Based},
  keywords = {Explainable recommendation, Recommender system, Graph Neural Network, Multi-task learning, Collaborative filtering},
  url      = {https://www.sciencedirect.com/science/article/pii/S0893608022004087},
}
```

## References

[^1]: [RecBole Recommendation Toolbox](https://recbole.io/)

[^2]: [EXTRA (EXplanaTion RAnking) Datasets](https://github.com/lileipisces/EXTRA)
