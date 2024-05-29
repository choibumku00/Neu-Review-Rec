# -*- encoding: utf-8 -*-
import time
import fire
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import ReviewData
from LitModel import LitModel
import config

# TensorFlow oneDNN 설정 변경
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class MetricsLogger(pl.callbacks.Callback):
    def on_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        val_mse = trainer.callback_metrics.get('val_mse')
        val_mae = trainer.callback_metrics.get('val_mae')
        
        print(f'\nEpoch {trainer.current_epoch}:')
        if train_loss is not None:
            print(f'  train_loss = {train_loss:.4f}')
        if val_mse is not None:
            print(f'  val_mse = {val_mse:.4f}')
        if val_mae is not None:
            print(f'  val_mae = {val_mae:.4f}')

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def run(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    pl.seed_everything(opt.seed)

    print(f"Running model: {opt.model}")  # 실행 중인 모델 이름 출력

    litModel = LitModel(opt)

    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=opt.num_workers,
                                   persistent_workers=True)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=opt.num_workers,
                                 persistent_workers=True)
    print(f'train data: {len(train_data)}; val data: {len(val_data)};')

    ckpt = ModelCheckpoint(dirpath='./checkpoints/', monitor='val_mse', mode='min',
                           filename=f"{opt.model}-" + "{epoch}-{val_mse:.4f}-{val_mae:.4f}")
    
    metrics_logger = MetricsLogger()
    
    # Tensor Core 최적화 설정
    torch.set_float32_matmul_precision('high')

    trainer = pl.Trainer(
        devices='auto',
        max_epochs=opt.num_epochs,
        accelerator='auto',
        callbacks=[ckpt, metrics_logger],
        precision='16-mixed',  # Mixed Precision Training 활성화
        enable_progress_bar=True
    )

    trainer.fit(litModel, train_data_loader, val_data_loader)


if __name__ == "__main__":
    fire.Fire()
