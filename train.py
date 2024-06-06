import argparse
import torch
import wandb

from trainer import build_trainer
from utils import collect_env_info, get_cfg_default, set_random_seed, setup_logger

def reset_cfg_from_wandb(cfg):
    cfg.DATALOADER.TRAIN.BATCH_SIZE = wandb.config.DATALOADER_TRAIN_BATCH_SIZE
    cfg.OPTIM.LR = wandb.config.OPTIM_LR
    # cfg.OPTIM.MAX_EPOCH = wandb.config.OPTIM_MAX_EPOCH
    cfg.MODEL.CLIPAdapters.DOMAIN_LOSS_WEIGHT = wandb.config.DOMAIN_LOSS_WEIGHT  
    # cfg.SEED = wandb.config.SEED

def reset_cfg_from_args(cfg, args):
    cfg.GPU = args.gpu
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = args.root

    if args.dataset:
        cfg.DATASET.NAME = args.dataset
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.model:
        cfg.MODEL.NAME = args.model

def clean_cfg(cfg, model):
    keys = list(cfg.MODEL.keys())
    for key in keys:
        if key == "NAME" or key == model:
            continue
        cfg.MODEL.pop(key, None)

def setup_cfg(args):
    cfg = get_cfg_default()

    if args.model_config_file:
        cfg.merge_from_file(args.model_config_file)

    reset_cfg_from_args(cfg, args)
    reset_cfg_from_wandb(cfg)

    clean_cfg(cfg, args.model)

    cfg.freeze()

    return cfg

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def main(args):
    # Initialize W&B
    wandb.init(project="clip-adapters", config=args)  
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        wandb.log({"seed": cfg.SEED})
        set_random_seed(cfg.SEED)

    torch.cuda.set_device(cfg.GPU)

    setup_logger(cfg.OUTPUT_DIR)

    print("*** Config ***")
    print_args(args, cfg)

    trainer = build_trainer(cfg)
    if args.model == "CLIPZeroShot":
        trainer.test()
    else:
        trainer.train()

    wandb.log({"accuracy": trainer.test()})

if __name__ == "__main__":
    sweep_configuration = {
        # 'method': 'bayes',
        'method': 'grid', 
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'DATALOADER_TRAIN_BATCH_SIZE': {
                'values': [32, 64, 128]
            },
            'OPTIM_LR': {
                'values': [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05]
            },
            'DOMAIN_LOSS_WEIGHT': {
                'values': [0.1, 0.2, 0.3, 0.4]
            }
            # ,
            # 'SEED': {
            #     'values': [134, 232, 607, 779, 995] 
            # }
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--output-dir", type=str, default="./output/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--source-domains", type=str, nargs="+", default=["photo", "art_painting", "cartoon"])
    parser.add_argument("--target-domains", type=str, nargs="+", default=["sketch"])
    parser.add_argument("--model", type=str, default="CLIPAdapters")  
    parser.add_argument("--model-config-file", type=str, default="config/clipadapters.yaml") 
    args = parser.parse_args()
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="clip-adapters")
    wandb.agent(sweep_id, function=lambda: main(args))
