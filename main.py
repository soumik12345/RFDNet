from rfdnet import Trainer
from rfdnet.utils import init_wandb


init_wandb(
    project_name='rfdnet', experiment_name='div2k_train',
    wandb_api_key='cf0947ccde62903d4df0742a58b8a54ca4c11673'
)

trainer = Trainer()
trainer.build_dataset(
    dataset_url='http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
    image_limiter=100
)
trainer.compile()
trainer.train()
