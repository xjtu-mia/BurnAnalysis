# BurnAnalysis
## 1. Environment
- Please prepare an environment with python=3.10|cudatoolkit=11.3.1, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare Burns datasets 

## 3. Train model 
```bash
python train_net.py --data-split=0, --batch-size=4, --max-epoch=150, --ckpt-dir='ckpt/tmpckpt'
```
## 4. Test model and evaluate
```bash
python test_eval.py --data-split=0, --batch-size=4, --ckpt-dir='ckpt/tmpckpt', --out-dir='output/tmp', --cndct-vis=False
```