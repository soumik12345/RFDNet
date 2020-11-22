import os
import wandb


def init_wandb(project_name, experiment_name, wandb_api_key):
    """Initialize Wandb
    Args:
        project_name: project name on Wandb
        experiment_name: experiment name on Wandb
        wandb_api_key: Wandb API Key
    """
    if project_name is not None and experiment_name is not None:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(project=project_name, name=experiment_name)


def lr_scheduler(epoch, lr):
    if epoch % 20:
        return lr / 2.0
    return lr
