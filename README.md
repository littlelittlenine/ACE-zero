# ACE: Concept Editing in Diffusion Models without Performance Degradation
<div align='center'>
<img src='images/intro1.png'>
</div>

Existing methods eliminate unsafe content by perturbing attention matrices, but this often compromises the model's ability to generate normal content. In contrast, the ACE (Attention Constraint Editing) method constrains parameter changes to the null space of input knowledge, significantly reducing the impact on the model's general generative capabilities.

## Installation Guide

The code base is based on the `diffusers` package. To get started:
```
git clone https://github.com/rohitgandikota/unified-concept-editing.git](https://github.com/littlelittlenine/ACE-zero.git
cd ACE-zero
mkdir models
pip install -r requirements.txt
```
### Erasing Artists
```
python /train-scripts/erase_nude_alphaedit.py --model_save_path /models/edit.pt --concepts_save_path /models/edit.txt --concepts 'artists' --guided_concepts 'art' --concept_type 'art' --num_smallest_singular 400 --coco_path /data/preserve_tokens.csv --lamda 100 --device 0
```
### Moderating
To moderate concepts (e.g. "violence, nudity, harm")
```
python /train-scripts/erase_nudity.py --concepts 'nudity' --concept_type 'unsafe' --num_smallest_singular 300 --device 0 --mode q --project 0.16 
```
### Debiasing
To debias concepts (e.g. "Doctor, Nurse, Carpenter") against attributes (e.g. "Male, Female") 
```
python /train-scripts/train_debias_nullspace.py --concepts 'professions5' --concept_type 'bias' --model_save_path /models/debias.pt --concepts_save_path /models/debias.txt --coco_path /data/preserve_tokens.csv --lamda 10

