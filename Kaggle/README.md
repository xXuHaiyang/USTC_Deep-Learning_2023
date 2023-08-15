# Kaggle icr Competition

## Get ready for environment
```
conda create -n icr python=3.9
conda activate icr
pip install -r requirements.txt
pip install ./pip-packgaes/tabpfn-0.1.9-py3-none-any.whl
mkdir -p ~/softwares/anaconda3/envs/icr/lib/python3.9/site-packages/tabpfn/models_diff/
cp ./ckpts/prior_diff_real_checkpoint_n_0_epoch_100.cpkt ~/softwares/anaconda3/envs/icr/lib/python3.9/site-packages/tabpfn/models_diff
```

## TODO
In `postprocessin-ensemble` we already have a powerful ensemble method, it's no need to modify it. We can focus on the following things:
0. Support GPU training. DONE. 15min -> 35s.
1. Add validation code on cross-validation data, and get the CV score to self-validate the score. DONE. Val acc is close to 1.0.
2. Preprocess the data by hand (normalize, see their inner-correlation, etc.)
3. Tuning the hyper-parameters of the model, e.g., 0/1 post-processing threshold, etc.

email | phone number | name | description
--- | --- | --- | ---
gmail | 131 | Haiyang Xu | 4 people
hax027 | 408 | Daniel Xu | 2 people 
xuhaiyang | 132 | Haiyang Daniel| 4 people 
4074 | 159 | Xu Haiyang | 2 people