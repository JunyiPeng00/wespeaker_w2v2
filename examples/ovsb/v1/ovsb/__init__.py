"""Recipe package of the OVSB speaker-verification recipe.

Layers, lowest first; a layer imports only the layers listed for it:

* ``ovsb.config``  - typed recipe config (stdlib and yaml only).
* ``ovsb.data``    - index, sampler, crop, augmentation, collate
  (``ovsb.config``, ``wespeaker.dataset``, numpy, torch).
* ``ovsb.model``   - frontend, classifier head, model builder
  (``ovsb.config``, ``wespeaker.frontend``, ``wespeaker.models``, torch).
* ``ovsb.losses``  - loss functions on tensors (torch only).
* ``ovsb.engine``  - training step, loop, checkpoints (all layers above,
  ``wespeaker.utils``).
* ``ovsb.bin``     - command-line entry points (any layer).
"""
