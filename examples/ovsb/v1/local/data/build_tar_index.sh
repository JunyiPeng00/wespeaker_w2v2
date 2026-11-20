#!/bin/bash

# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tar index of the shards of one corpus (random access for the sampler and
# an input of local/data/build_train_list.py). Run from the recipe
# directory:
#
#   bash local/data/build_tar_index.sh --shard_list <corpus>/shard.list \
#     --out_dir <corpus>/tar_index [--nj 4] [--root <dir>]
#
# --root: directory for relative shard paths (default: this directory). The
# options are those of `python -m ovsb.data.index build`. CPU only; one
# 2 KiB read per utterance.

. ./path.sh
set -eo pipefail

if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  exec python -m ovsb.data.index build --help
fi
exec python -m ovsb.data.index build "$@"
