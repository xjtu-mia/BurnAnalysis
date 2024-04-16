# BurnAnalysis
## 1. Environment
- Please prepare an environment with python=3.10|cudatoolkit=11.3.1, and then use the command "pip install -r requirements.txt" for the dependencies.
## 2. Prepare Burns datasets 

## 3. Train model 
- To train a multi-task network, you can enter the following command in the terminal interface.
```bash
python train_net.py --data-split='0', --batch-size=4, --max-epoch=150, --ckpt-dir='ckpt/tmpckpt'
```
## 4. Test model and evaluate
- To test a trained multi-task network, you can enter the following command in the terminal interface.
```bash
python test_eval.py --data-split='0', --batch-size=4, --ckpt-dir='ckpt/tmpckpt', --out-dir='output/tmp', --cndct-vis=False
```

## BurnAI Application (Android OS version ≥10)
- You can download the installation pakage from the [Baidu Netdisk](https://pan.baidu.com/s/1aoKYvQDE6wu8gvTyfAMCkw?pwd=gr67 ) (access code: gr67). 