## Datasets

```
Please download the datasets [here](https://drive.google.com/file/d/1uJ2omzIs0NCtJsGQsyFCBHCXUhoK1mkO/view?usp=sharing) and extract them into root directory.
ent_ids_1: entity ids in the source KG
ent_ids_2: entity ids in the target KG
ref_ent_ids: entity alignment, list of pairs like (e_s \t e_t)
triples_1: relation triples in the source KG
triples_2: relation triples in the target KG
```
## Environment

```
apex
pytorch
torch_geometric
```

## Running

```
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang zh_en
```
