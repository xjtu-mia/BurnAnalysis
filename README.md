# BurnAnalysis
## 1. Environment
- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare Burns datasets 
----------------------------------. 
## 2. Train model 
```bash
python train_net.py --base-lr=1e-4, --data-split=0, --batch-size=4, --max-epoch=150, --ckpt-dir=ckpt/tmpckpt, --backbone='convnext
```