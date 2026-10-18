# Copyright (c) 2024 Hongji Wang (jijijiang77@gmail.com)
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


from .w2vbert import W2VBertFrontend
from .tfmel import TFMelFrontend

# s3prl / whisper are optional dependencies; keep the corresponding
# frontends available only when the packages are installed.
try:
    from .s3prl import S3prlFrontend
except ImportError:
    S3prlFrontend = None
try:
    from .whisper_encoder import whisper_encoder
except ImportError:
    whisper_encoder = None


def _w2vbert_sv_frontend(*args, **kwargs):
    # Lazy factory: only import transformers' Wav2Vec2BertModel when this
    # frontend is requested.
    from .w2vbert_sv import W2VBertSVFrontend
    return W2VBertSVFrontend(*args, **kwargs)


def _omniasr_sv_frontend(*args, **kwargs):
    from .omniasr_sv import OmniASRSVFrontend
    return OmniASRSVFrontend(*args, **kwargs)


frontend_class_dict = {
    'fbank': None,
    's3prl': S3prlFrontend,
    'whisper_encoder': whisper_encoder,
    'w2vbert': W2VBertFrontend,
    'tfmel': TFMelFrontend,
    'w2vbert_sv': _w2vbert_sv_frontend,
    'omniasr_sv': _omniasr_sv_frontend,
}
