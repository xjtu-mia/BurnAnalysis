import os
import shutil
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from modeling import MTNet, add_coswin, DynamicTaskWeightAlign
from evaluate import SemanticEvaluator, InstanceEvaluator
from dataset import get_data_loader
from detectron2.utils.logger import setup_logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traning script.")
    parser.add_argument('--base-lr', help='Initial learning rate', default=1e-4)
    parser.add_argument('--data-split', help='Data split index', default='0')
    parser.add_argument('--batch-size', help='Batch size', default=4)
    parser.add_argument('--max-epoch', help='Max epoch', default=150)
    parser.add_argument('--ckpt-dir', help='Checkpoint save directory', default='ckpt/tmp')
    parser.add_argument('--backbone', help='Backbone network', default='convnext') 
    parser.add_argument('--part-decoder', help='Body part decoder', default='Pointrend')
    parser.add_argument('--burn-decoder', help='Burn region decoder', default='BurnDecode')
    parser.add_argument('--part-num-classes', help='Number of body parts classes', default=11)
    parser.add_argument('--burn-num-classes', help='Number of burn degrees classes', default=4)
    parser.add_argument('--burn-class-weights', help='Class weights of burn degrees', default=[1, 1, 2, 4])

    args = parser.parse_args()
    BASE_LR = args.base_lr
    MAX_EPOCH = args.max_epoch
    BATCH_SIZE = args.batch_size
    CKPT_DIR = args.ckpt_dir
    BACKBONE_NAME = args.backbone

    data_cfg = {'split' : args.data_split,
            'longest_max_size': [384, 448, 512],
            'batch_size': BATCH_SIZE,
            }

    if BACKBONE_NAME == 'convnext':
        backbone_cfg = {'name': 'convnext',
                'arch': 'base',
                'depth': '',
                 'out_channels': [128, 256, 512, 1024],
                }
    elif BACKBONE_NAME == 'swin':
                backbone_cfg = {'name': 'swin',
                'arch': 'base',
                'depth': '',
                 'out_channels': [128, 256, 512, 1024], 
                }
    elif BACKBONE_NAME == 'mpvit':
                backbone_cfg = {'name': 'mpvit',
                'arch': 'base',
                'depth': '',
                 'out_channels': [224, 368, 480, 480], 
                }
    elif BACKBONE_NAME == 'maxvit':
                backbone_cfg = {'name': 'maxvit',
                'arch': 'small',
                'depth': '',
                 'out_channels': [96, 192, 384, 768], 
                }
    elif BACKBONE_NAME == 'resnet':
                backbone_cfg = {'name': 'maxvit',
                'arch': '',
                'depth': 152,
                 'out_channels': [256, 512, 1024, 2048], 
                }
    elif BACKBONE_NAME == 'resnext':
                backbone_cfg = {'name': 'resnext',
                'arch': '',
                'depth': 152,
                 'out_channels': [256, 512, 1024, 2048], 
                }
    instance_decoder_cfg = {'name': args.part_decoder, 
                            'num_classes': args.part_num_classes}
    burn_decoder_cfg = {'name': args.burn_decoder,
                    'num_classes': args.burn_num_classes,
                    'class_weights': args.burn_class_weights,
                    }

    model = MTNet(backbone_cfg, instance_decoder_cfg, burn_decoder_cfg)
    model = add_coswin(model, backbone_cfg['out_channels'],
                   window_size=7, att_type='NonLocal')
    dtwa = DynamicTaskWeightAlign()

    param_groups = [
                {'params': model.parameters()},
                {'params': dtwa.parameters()}]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device=device)

    logger = setup_logger(name='burnseg')
    logger.info(model)
    train_loader, test_loader = get_data_loader(split=data_cfg['split'],
                                                batch_size=data_cfg['batch_size'],
                                                longest_max_size=data_cfg['longest_max_size'])

    optimizer = torch.optim.AdamW(param_groups, 
                                BASE_LR, betas=[0.9, 0.99], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, gamma=0.5, milestones=[4000, 8000, 12000, 16000])

    log_dir = f"{CKPT_DIR}/log"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    # Note: 每次评估需要清空上次结果  evaluator.reset()
    evaluators = [InstanceEvaluator(), SemanticEvaluator()]
    cur_iter, cur_epoch = 0, 0
    cur_val_loss = float('inf')
    cur_all_metrics = {}
    trigger_times, patience = 0, 4 # trigger times and patience, arguments for 'early stopping'
    loss_dict_total = {} # 记录20个iteration的总损失
    while cur_epoch <= MAX_EPOCH and trigger_times < patience:
        #--------------------train----------------------
        cur_epoch += 1
        model.train()
        for batched_input in train_loader:
            cur_iter += 1
            loss_dict = model(batched_input)  # forward to calculate loss
            loss_dict, sigmas = dtwa(loss_dict)
            # loss = sum(loss_dict.values())
            # loss = 1/2σ1**2 loss1 + 1/2σ2**2 loss2 + log(σ1σ2)
            loss = sum(loss_dict.values()) + sum(torch.log(sigmas))
            loss_dict.update({'total_loss': loss})

            loss.backward()  # 反向传播,计算梯度
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()  # 梯度归零

            loss_dict_total = {k : loss_dict[k] + loss_dict_total.get(k, 0) 
                            for k in loss_dict.keys()}  # accumulate loss items

            if cur_iter % 20 == 0: # 每20次iteration记录一次loss
                cur_lr = optimizer.param_groups[0]["lr"]
                loss_dict_avg = {k:v/20 for k, v in loss_dict_total.items()}
                sigmas = {f"sigma{i}" : sigmas[i] for i in range(len(sigmas))}
                writer.add_scalars('sigmas', sigmas, cur_iter) # 记录学习权重
                writer.add_scalar('lr', cur_lr, cur_iter) # record learning rate
                writer.add_scalars('loss', loss_dict, cur_iter) # record loss 
                msg = f"epoch: {cur_epoch} iter: {cur_iter} lr: {cur_lr} " + \
                    ' '.join([f"{k}: {v :.5f}" for k,
                            v in loss_dict_avg.items()])
                logger.info(msg)
                loss_dict_total = {}   
            # 每隔2000次迭代保存模型参数
            # if cur_iter % 2000 == 0:
            #     torch.save(model.state_dict(), f"{CKPT_DIR}/iter_{cur_iter}.pth")
        # 每3个epoch记录下验证损失
        if cur_epoch % 3 == 0:
            loss_dict_total_val = {}
            batch_counter = 0
            model.eval()
            with torch.no_grad():
                for batched_input in test_loader:
                    batch_counter += 1
                    loss_dict = model(batched_input)  # forward to calculate loss
                    loss = sum(loss_dict.values())
                    loss_dict.update({'total_loss': loss})
                    loss_dict_total_val = {k : loss_dict[k] + loss_dict_total_val.get(k, 0) 
                                for k in loss_dict.keys()}  # accumulate loss items
                # 记录验证损失
                loss_dict_avg_val = {k:v/batch_counter for k, v in loss_dict_total_val.items()}
                writer.add_scalar('val_loss', loss_dict_avg_val['total_loss'], cur_epoch)
                # 记录评估指标
                all_metrics = {}
                for evaluator in evaluators:
                    evaluator.reset() # 清空evaluator缓存的结果
                for batched_input in test_loader:
                    preds = model.inference(batched_input)  # forward to inference
                    for evaluator in evaluators:
                        evaluator.process(batched_input, preds)
                for evaluator in evaluators:
                    metrics = evaluator.evaluate()
                    all_metrics.update(metrics)
                logger.info(all_metrics)
                # 早停策略
                # if loss_dict_avg_val['total_loss'] <= cur_val_loss:
                #     cur_val_loss = loss_dict_avg_val['total_loss']
                #     trigger_times = 0
                #     torch.save(model, f"{CKPT_DIR}/best_model.pth")
                # else:
                #     trigger_times += 1

                # 比较性能
                ap, cur_ap = all_metrics['AP'], cur_all_metrics.get('AP', 0)
                dice, cur_dice =  all_metrics['Dice'], cur_all_metrics.get('Dice', [0] * 2)
                # 不计算背景类的dice
                mdice, cur_mdice = sum(dice[1:]) / len(dice[1:]), \
                                    sum(cur_dice[1:]) / len(cur_dice[1:])
                writer.add_scalar('ap', ap, cur_epoch)
                writer.add_scalar('mdice', mdice, cur_epoch)
                if (ap + mdice) - (cur_ap + cur_mdice) >= 0.2:
                    cur_all_metrics = all_metrics
                    trigger_times = 0
                    torch.save(model, f"{CKPT_DIR}/best_model.pth")
                else:
                    trigger_times += 1