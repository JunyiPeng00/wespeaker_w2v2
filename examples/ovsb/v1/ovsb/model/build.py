"""Assemble the speaker model: pooling backend, encoder, classifier head.

The model is the wespeaker pooling backend with two modules added by name:

    <backend parameters>      weights_k, weights_v, cmp_linear_*, norm_*, ...
    frontend.*                 the (truncated) W2V-BERT 2.0 encoder
    projection.weight          the sub-centre classifier head

so its state dict has the keys and shapes of a model built by the stock
wespeaker trainer and checkpoints are interchangeable with it.

The construction order is fixed: encoder, backend, truncation, head. It
decides the parameter order (backend, encoder, head), which the optimiser
param groups follow, and the order in which the random initialisations
consume the global torch generator.
"""
import logging
from typing import Optional

import torch

from ovsb.config.core_view import FRONTEND, to_core_dict
from ovsb.config.schema import Config
from ovsb.model.frontend import TimeMaskedW2VBertSV
from ovsb.model.head import DurationMarginSubcenterHead
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.frontend_utils import maybe_truncate_frontend

FRONTEND_MODULE = 'frontend'
HEAD_MODULE = 'projection'


def build_model(cfg: Config, num_classes: int,
                logger: Optional[logging.Logger] = None) -> torch.nn.Module:
    """Build the model of ``cfg`` with a head for ``num_classes`` speakers.

    Args:
        cfg: the recipe config (``model`` and ``data.resample_rate``).
        num_classes: number of speakers of the training list.
        logger: optional logger for the truncation message.

    Returns:
        The backend module with ``frontend`` and ``projection`` attached.
        ``model.frontend(wav, wav_len)`` gives the layer features,
        ``model(features, frame_lens)`` the embeddings and
        ``model.projection(emb, label, margin_scale)`` the margin logits.
        The head carries the margin ``cfg.model.head.margin``.

    Raises:
        ValueError: ``model.backend.feat_dim`` is not the encoder width.
    """
    core = to_core_dict(cfg)
    data_args = core['dataset_args']
    time_mask = cfg.model.frontend.time_mask
    frontend = TimeMaskedW2VBertSV(
        **data_args[FRONTEND + '_args'],
        sample_rate=cfg.data.resample_rate,
        time_mask={'prob': time_mask.prob, 'span': time_mask.span})
    if frontend.output_size() != cfg.model.backend.feat_dim:
        raise ValueError(
            'model.backend.feat_dim: %d, but the encoder in %s has width %d'
            % (cfg.model.backend.feat_dim, cfg.model.frontend.assets,
               frontend.output_size()))
    model = get_speaker_model(core['model'])(**core['model_args'])
    model.add_module(FRONTEND_MODULE, frontend)
    maybe_truncate_frontend(model, data_args, cfg.model.backend.nb_layer,
                            logger)
    head_cfg = cfg.model.head
    head = DurationMarginSubcenterHead(
        cfg.model.backend.embed_dim, num_classes, scale=head_cfg.scale,
        margin=0.0, easy_margin=False, K=head_cfg.K, mp=head_cfg.mp,
        k_top=head_cfg.k_top)
    # update() also sets the hard-negative penalty, which the constructor
    # leaves at zero
    head.update(head_cfg.margin)
    model.add_module(HEAD_MODULE, head)
    return model
