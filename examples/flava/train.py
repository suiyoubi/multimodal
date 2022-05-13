# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from callbacks.multimodal_eval import MultimodalEvalCallback
from data import (
    ImageDataModule,
    ImageDataModuleOld,
    MLMDataModule,
    MultiDataModule,
    VLDataModule,
)
from definitions import FLAVAArguments
from model import FLAVAPreTrainingLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import build_config, build_datamodule_kwargs
from datetime import timedelta
import os

def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    datamodules = []
    # imagenet_datamodule = ImageDataModule(
    #     **build_datamodule_kwargs(config.datasets.image, config.training)
    # )
    imagenet_datamodule = ImageDataModuleOld(
        train_root=config.image_folder.train_root,
        val_root=config.image_folder.val_root,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        allow_unenven_batchs=False,
    )
    if "image" in config.datasets.selected:
        datamodules.append(imagenet_datamodule)

    if "text" in config.datasets.selected:
        mlm_datamodule = MLMDataModule(
            **build_datamodule_kwargs(config.datasets.text, config.training)
        )
        datamodules.append(mlm_datamodule)

    if "vl" in config.datasets.selected:
        vl_datamodule = VLDataModule(
            **build_datamodule_kwargs(config.datasets.vl, config.training)
        )
        datamodules.append(vl_datamodule)

    datamodule = MultiDataModule(datamodules)

    datamodule.setup("fit")
    model = FLAVAPreTrainingLightningModule(
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        **config.model,
    )

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule),
            ModelCheckpoint(
                filename="{epoch}-{step}",
                train_time_interval=timedelta(minutes=1),
                save_last=True,
                save_top_k = -1
            )
        ],
        strategy="ddp",
    )

    prev_ckpt = os.path.join(
        config.training.lightning['default_root_dir'], 
        'lightning_logs', 
        f'version_{config.training.prev_v_num}', 
        'checkpoints', 
        'last.ckpt'
        )
    if os.path.exists(prev_ckpt):
        print(f'Resuming from last checkpoint: {prev_ckpt}')
    else:
        prev_ckpt = None

    trainer.fit(
        model, datamodule=datamodule,
        ckpt_path=prev_ckpt
    )
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
