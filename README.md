# MR-DTR

Time-aware Medication Recommendation via Intervention of Dynamic Treatment Regimes


## Environment
```python
python==3.9.18
torch==2.1.1
tqdm==4.66.1
dgl==1.1.2.cu118
scikit-learn==1.3.2
```
You can build the conda environment for our experiment using the following command:
```
conda env create -f environment.yml
```



## Prepare the datasets

We used two datasets, [MIMIC-III](https://mimic.mit.edu/docs/iii/) v1.4 and [MIMIC-IV](https://mimic.mit.edu/docs/iv/) v1.0, for our experiments.

The files required is the same as [SafeDrug](https://github.com/ycq091044/SafeDrug/), and the SMILES file needs to be processed as provided in SafeDrug using DrugBank.

If the above conditions are ready, run the following commands in sequence to preprocess:

```python
# preprocess the dataset(mimic-iii)
cd data
python preprocess_mimic-iii.py
# preprocess the dataset(mimic-iv)
python preprocess_mimic-iv.py

# preprocess the co-guided temporal graph
cd ..
python preprocess_graph.py

```



## Train or Test


You can train or test the model using the following command:
```python
# mimic-iii
python main_mimic-iii.py
python main_mimic-iii.py --Test
# mimic-iv
python main_mimic-iv.py
python main_mimic-iv.py --Test
```

## Acknowledgement
None
