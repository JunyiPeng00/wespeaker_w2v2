"""The recipe config in the schema of the stock wespeaker tools.

``wespeaker/bin/extract.py`` and ``average_model.py`` read a wespeaker
config (``model``, ``model_args``, ``dataset_args``). :func:`to_core_dict`
writes the part of the recipe config they need, so embeddings are extracted
with the unmodified tools. Training-only options (frame masking, sampler,
augmentation) are not part of this view.
"""

from typing import Any, Dict

from ovsb.config.schema import Config

FRONTEND = 'w2vbert_sv'
# 20 ms frames: 10 ms fbank hop, stride-2 stacking in the frontend
FRAME_MS = 20


def to_core_dict(cfg: Config) -> Dict[str, Any]:
    """wespeaker-schema dict for extraction and model averaging."""
    backend, frontend = cfg.model.backend, cfg.model.frontend
    return {
        'model': backend.name,
        'model_args': {
            'feat_dim': backend.feat_dim,
            'head_nb': backend.head_nb,
            'embed_dim': backend.embed_dim,
            'compression_dim': backend.compression_dim,
            'feature_grad_mult': backend.feature_grad_mult,
            'nb_layer': backend.nb_layer,
        },
        'dataset_args': {
            'frontend': FRONTEND,
            'frontend_num_layers': frontend.num_layers,
            FRONTEND + '_args': {
                'upstream_args': {'path_or_url': frontend.assets},
                'feature_extractor': 'torch',
                'frozen': False,
                'gradient_checkpointing': frontend.gradient_checkpointing,
                'frame_shift': FRAME_MS,
                'frame_length': FRAME_MS,
            },
            'resample_rate': cfg.data.resample_rate,
            'num_frms': cfg.data.num_frms,
            'cmvn': False,
        },
    }
