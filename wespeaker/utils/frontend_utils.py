# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frontend helpers that work from outside wespeaker/frontend/*."""

import torch.nn as nn


def truncate_frontend_layers(frontend, num_layers):
    """Keep only the first `num_layers` encoder layers of an SSL frontend.

    The W2V-BERT frontend (`w2vbert_sv.W2VBertSVFrontend`) iterates over
    `frontend.upstream.encoder.layers` and return the input of layer 1 plus
    every layer output, so after truncation the frontend runs `num_layers`
    layers and returns `num_layers + 1` hidden states (MHFA nb_layer must be
    num_layers + 1). The dropped layers are deleted (no unused parameters
    under DDP, ~12 % less encoder compute for 21 of 24 layers).

    Returns the number of dropped layers.
    """
    encoder = frontend.upstream.encoder
    layers = list(encoder.layers)
    assert 0 < num_layers <= len(layers), (num_layers, len(layers))
    encoder.layers = nn.ModuleList(layers[:num_layers])
    config = getattr(frontend.upstream, 'config', None)
    if config is not None and hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = num_layers
    if hasattr(frontend, 'n_hidden_states'):
        frontend.n_hidden_states = num_layers + 1
    return len(layers) - num_layers


def maybe_truncate_frontend(model, dataset_args, nb_layer=None, logger=None):
    """Apply dataset_args['frontend_num_layers'] (None/absent = no-op)."""
    num_layers = dataset_args.get('frontend_num_layers', None)
    if not num_layers:
        return 0
    dropped = truncate_frontend_layers(model.frontend, int(num_layers))
    if nb_layer is not None:
        assert nb_layer == int(num_layers) + 1, (
            'model_args.nb_layer ({}) must be frontend_num_layers + 1 ({})'.
            format(nb_layer, int(num_layers) + 1))
    if logger is not None:
        logger.info('frontend truncated to {} encoder layers ({} dropped)'.
                    format(num_layers, dropped))
    return dropped
