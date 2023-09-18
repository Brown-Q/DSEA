## DSEA
Code for ESWA2023 paper [“Leveraging multimodal features for knowledge graph entity alignment based on dynamic self-attention networks”](https:10.1016/j.eswa.2023.120363).

## Datasets
Please download the datasets [here](https://drive.google.com/file/d/1Q1xqwpWnqjg3X5unmXNfBz_xqVqVdy_E/view?usp=share_link) and extract them into root directory.  
ent_ids_1: entity ids in the source KG  
ent_ids_2: entity ids in the target KG  
ref_ent_ids: entity alignment, list of pairs like (e_s \t e_t)  
triples_1: relation triples in the source KG  
triples_2: relation triples in the target KG  

## Environment
python==3.6.13  
networkx==2.5.1  
apex==0.1  
pytorch==1.7.1  
torch_geometric==2.0.2  
cuda==11.2     

## Running  
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang ja_en  

## Citation
If you use this model or code, please cite it as follows:  
@article{qian2023leveraging,  
  title={Leveraging multimodal features for knowledge graph entity alignment based on dynamic self-attention networks},  
  author={Qian, Ye and Pan, Li},  
  journal={Expert Systems with Applications},  
  volume={228},  
  pages={120363},  
  year={2023},  
  publisher={Elsevier}  
}  
