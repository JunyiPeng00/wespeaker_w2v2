# Copyright (c) 2024 Junyi Peng (pengjy@fit.vut.cz)
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

"""Frontend for speech feature extraction using Hugging Face pretrained models."""

import contextlib
import os
from typing import Mapping, Any, Tuple, Optional

import json
import torch
import torch.nn as nn
import torch.distributed as dist

from transformers import AutoFeatureExtractor, AutoModel
from wespeaker.frontend.wav2vecbert.configuration_wav2vec2_bert import Wav2Vec2BertConfig
from wespeaker.frontend.wav2vecbert.modeling_wav2vec2_bert import Wav2Vec2BertModel
from wespeaker.frontend.wav2vecbert.forward_impl import forward_w2v_bert

def is_rank_zero() -> bool:
    """Check if current process is rank 0 in distributed training.
    
    Returns:
        True if current process is rank 0 or not in distributed training.
    """
    # First check environment variable (for manual testing)
    rank_env = os.environ.get('RANK')
    if rank_env is not None:
        return int(rank_env) == 0
    
    # Then check distributed training
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True  # If not distributed, always log
    return torch.distributed.get_rank() == 0


class HuggingfaceFrontend_w2vbert(nn.Module):
    """Wraps a Hugging Face pretrained model for speech feature extraction.

    This module handles the download, conversion, and loading of a specified
    Hugging Face WavLM/XLS-R model. It serves as a feature extractor frontend
    and supports optional freezing of weights and model pruning.

    Attributes:
        upstream_name: The name of the upstream model (e.g., 'wavlm_large').
        frozen: A boolean indicating if the upstream model weights are frozen.
        upstream: The loaded pretrained model instance.
        upstream_config: The configuration dictionary of the upstream model.
    """

    # Maximum input length in samples (120 seconds * 16kHz)
    _MAX_INPUT_SAMPLES = 10 * 16000

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        download_dir: str = './hf_models/',
        frozen: bool = False,
        frame_shift: int = 20,
        frame_length: int = 20,
        sample_rate: int = 16000,
    ):
        """Initializes the HuggingfaceFrontend.

        Args:
            upstream_args: A dictionary containing configuration for the upstream
                model. Must include 'name' (e.g., 'wavlm_base_plus'). It can
                also include 'pruning_units'.
            download_dir: The directory to save downloaded and converted
                Hugging Face models.
            frozen: If True, the model parameters are frozen and not trained.
        """
        super().__init__()
        # if 'prune' in upstream_args:
        self.model_config = upstream_args['name'].lower()
        # else:
        #     self.model_config = None
        # self.model_config = upstream_args['name'].lower() if upstream_args['name'] is not None else None
        download_dir = upstream_args['path_or_url']
        self.frozen = frozen

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            download_dir,
            local_files_only=True,
            )

        self._setup_model(download_dir)

        # 3. Freeze weights if required.
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
            self.upstream.eval()
        else:
            for param in self.upstream.parameters():
                param.requires_grad_(True)

    def _setup_model(self, local_model_path):
        if self.model_config is not None:
            with open(os.path.join(local_model_path,self.model_config), 'r') as f:
                config_dict = json.load(f)
            config = Wav2Vec2BertConfig(**config_dict)
            full_model = Wav2Vec2BertModel(config)
            if 'prune' not in self.model_config:
                from safetensors.torch import load_file
                ckpt_state_dict = load_file(os.path.join(local_model_path,'model.safetensors'), device='cpu')
                cur_state_dict = full_model.state_dict()
                for k in cur_state_dict.keys():
                    if 'hard_concrete' in k:
                        continue
                    if k in ckpt_state_dict and cur_state_dict[k].shape == ckpt_state_dict[k].shape:
                        cur_state_dict[k] = ckpt_state_dict[k]
                    else:
                        print(f'{k}_is_mismatch')
                full_model.load_state_dict(cur_state_dict)
        else:
            full_model = AutoModel.from_pretrained(
                local_model_path,
                local_files_only=True,
                )

        self.upstream = full_model
        self.d_model = self.upstream.config.hidden_size
        self.n_hidden_states = self.upstream.config.num_hidden_layers + 1
        # 删除微调时不需要的masked_spec_embed参数，确保DDP兼容性
        delattr(self.upstream, 'masked_spec_embed')


    def output_size(self) -> int:
        """Returns the output feature dimension of the model.

        Raises:
            ValueError: If the model name is unknown.

        Returns:
            The integer size of the output dimension.
        """

        return self.d_model 
                

    def forward(
        self, input_wav: torch.Tensor, input_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Performs the forward pass to extract features.

        Args:
            input_wav: A batch of input waveforms, shape (B, T).
            input_lengths: A batch of waveform lengths, shape (B,). This
                argument is unused but maintained for API compatibility.

        Returns:
            A tuple containing:
                - The extracted features, shape (B, D, F, L), where D is the
                  feature dimension, F is the number of frames, and L is the
                  number of layers.
                - None, for API compatibility.
        """
        # Ensure model is not running on excessively long inputs
        input_tensor = input_wav[:, : self._MAX_INPUT_SAMPLES]

        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            # ssl_hiddens is a tuple of tensors, one for each layer
            out = forward_w2v_bert(self, input_tensor)

        # Stack layer representations and reorder dimensions
        # Original: (L, B, F, D) -> Stacked: (L, B, F, D)
        # Permuted: (B, D, F, L) for downstream convenience
        layer_reps = torch.stack(out.hidden_states, dim=0).permute(1, 3, 2, 0)
        return layer_reps, None

    def get_num_params(self) -> int:
        """Returns the total number of parameters in the upstream model."""
        if self.model_config is not None:
            return self.upstream.get_num_params()
        else:
            total_params = 0
            for param in self.upstream.parameters():
                total_params += param.numel()
            return total_params
    
    def prune(self) -> nn.Module:
        """Applies pruning to the upstream model."""
        return self.upstream.prune()


def main():
    """Instantiates and tests the HuggingfaceFrontend."""
    # Check if we're in distributed training
    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # This example config demonstrates support for various SSL models.
    # The conversion logic will handle downloading and converting the model.
    upstream_config = {
        'name': 'config_prune.json',  # Can be: hubert_base, wav2vec2_base, wavlm_base, etc.
        'path_or_url': "<path> original/examples/voxceleb/original recipe/convert_w2vbert/w2v-bert-2.0",
    }
    
    if is_rank_zero():
        print(f'Initializing HuggingfaceFrontend with config: {upstream_config}')
        if world_size > 1:
            print(f'Running in distributed mode: rank={rank}, world_size={world_size}')

    # Note: This will trigger the download and conversion of the specified model
    # if it's not already present in './hf_models/'.
    net = HuggingfaceFrontend_w2vbert(
        upstream_args=upstream_config, download_dir='./hf_models/'
    )

    dummy_input = torch.randn(4, 32000)  # Batch of 4, 2 seconds of audio
    dummy_lengths = torch.LongTensor([32000] * 4)

    if is_rank_zero():
        print('Model initialized successfully.')
        print(f'Number of parameters: {net.get_num_params():,}')
    print(net)  # Uncomment to see model architecture

    if is_rank_zero():
        print('Testing forward pass...')
    
    
    output, _ = net(dummy_input, dummy_lengths)
    print(f'Number of parameters before pruning: {net.get_num_params():,}')

    # from deepspeed.profiling.flops_profiler import get_model_profile
    
    # flops, macs, params  = get_model_profile(
    #     model=net.eval(),
    #     input_shape=(1, 16000*4),  # 根据你模型的输入更改
    #     print_profile=True,
    #     detailed=True,                 # 打印每一层
    #     module_depth=-1,               # 控制打印深度
    # )
    # print(f"flops: {flops}, macs; {macs}, Params: {params}")

    if is_rank_zero():
        print(f'Output shape: {output.shape}')


if __name__ == '__main__':
    main()