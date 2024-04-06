import argparse
import os
import torch
from dataset import get_data_loader
from evaluate import InstanceEvaluator, SemanticEvaluator
from detectron2.utils.logger import setup_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing&evaluating script.")
    parser.add_argument('--data-split', help='Data split index', default='0')
    parser.add_argument('--batch-size', help='Batch size', default=4)
    parser.add_argument('--ckpt-dir', help='Checkpoint save directory', default='ckpt/tmp')
    parser.add_argument('--out-dir', help='Evaluation result save directory', default='output/tmp')
    parser.add_argument('--cndct-vis', help='Whether conduct visualization', default=True)
    parser.add_argument('--part-vis-alpha', help='The alpha of body parts segmentation mask', default=0.5)
    parser.add_argument('--burn-vis-alpha', help='The alpha of burn regions segmentation mask', default=0.5)
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    evaluators = [InstanceEvaluator(output_dir=args.out_dir, 
                                    visualize=args.cndct_vis, 
                                    save_metrics=True, 
                                    alpha=args.part_vis_alpha, 
                                    viz_box=True),
                    SemanticEvaluator(output_dir=args.out_dir, 
                                      visualize=args.cndct_vis, 
                                      save_metrics=True,
                                      alpha=args.burn_vis_alpha)]

    model = torch.load(f"{args.ckpt_dir}/best_model.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger = setup_logger(name='burnseg')
    logger.info(model)
    train_loader, test_loader = get_data_loader(split=args.data_split, 
                                                batch_size=args.batch_size, 
                                                prefix=['train', 'test'])
    # evaluate
    model.eval()
    all_metrics = {}

    with torch.no_grad():
        for batched_input in test_loader:
            preds = model.inference(batched_input)  # forward to inference
            for evaluator in evaluators:
                evaluator.process(batched_input, preds)
        for evaluator in evaluators:
            metrics = evaluator.evaluate()
            all_metrics.update(metrics)
    print(all_metrics)