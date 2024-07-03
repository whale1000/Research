import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import logging
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    pl.seed_everything(opt.seed)  # 设置随机种子
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/",
        filename="epoch={epoch:04d}-val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        save_last=True,
        **opt.checkpoint
    )  # 模型检查点
    lr_monitor = pl.callbacks.LearningRateMonitor()  # 学习率监视器

    pl_logger = TensorBoardLogger("tensorboard", name="default", version=0)
    pl_profiler = AdvancedProfiler("profiler", "advance_profiler")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)  # 使用hydra库封装数据模块，opt.dataset表示配置文件中设置的数据集路径
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)  # 使用hydra库封装网络模块，opt.model表示配置文件中设置的模型
    trainer = pl.Trainer(gpus=1,
                         accelerator="gpu",
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0,  # disable sanity check
                         logger=pl_logger,
                        #  gradient_clip_val=0.1,
                        #  profiler=pl_profiler,
                         **opt.train)  # 使用pytorch_lighting封装了一个训练器

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))  # 获取检查点路径
    if len(checkpoints) > 0 and opt.resume:  # 当检查点大于0，且继续训练时
        print("Resume from", checkpoints[-1])
        trainer.fit(model, ckpt_path=checkpoints[-1])  # 加载检查点继续训练
    else:
        print("Saving configs.")
        OmegaConf.save(opt, "config.yaml")  # 保存配置文件
        trainer.fit(model)  # 开始训练


if __name__ == "__main__":
    main()
