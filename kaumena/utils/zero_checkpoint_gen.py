import os.path
from os import mkdir
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from typing import Dict
from ..models._mdxnet import Mixer, ConvTDFNet # noqa

def generate_zero_checkpoint(model: str, config: any, base_path="weights"):
    path = base_path + f'/{model}'
    match model:
        case 'mdxnet':
            return _gen_zero_for_mdxnet(config, path)
    return None


def _gen_zero_for_mdxnet(config:Dict, base_path: str) -> Dict[str, str]:
    sources = config.sources
    paths = {}
    for source in sources:
        source_config = OmegaConf.load(config.config_paths[source])
        model = ConvTDFNet(**{key: source_config[key] for key in dict(source_config) if key !='_target_'})
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer_states": [optimizer.state_dict() for optimizer in model.configure_optimizers()],
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": pl.__version__,
            "callbacks": {},
            "hparams_name": "hyper_parameters",
            "hyper_parameters": {},
        }
        if not os.path.exists(base_path):
            mkdir(base_path)
        torch.save(checkpoint, base_path + f'/{source}.ckpt')
        paths[source] = (base_path + f'/{source}.ckpt')

    mixer_config = OmegaConf.load(config.config_paths['mixer'])
    configs_without_separator = {key: config.config_paths[key] for key in dict(config.config_paths) if key !='mixer'}
    model = Mixer(separator_ckpts=paths, separator_configs=configs_without_separator, **{key: mixer_config[key] for key in dict(mixer_config) if key !='_target_' and key != 'separator_ckpts'})
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_states": [optimizer.state_dict() for optimizer in model.configure_optimizers()],
        "epoch": 0,
        "global_step": 0, "pytorch-lightning_version": pl.__version__,
        "callbacks": {},
        "hparams_name": "hyper_parameters",
        "hyper_parameters": {}
    }
    torch.save(checkpoint, base_path + f'/mixer.ckpt')
    paths["mixer"] = (base_path + f'/mixer.ckpt')
    return paths
