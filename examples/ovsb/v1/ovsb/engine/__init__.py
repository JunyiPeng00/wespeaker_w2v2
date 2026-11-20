"""Training engine: step, loop, optimiser, checkpoints, signals, logging.

Responsibility: one training step on a batch, the epoch loop, optimiser and
schedulers, checkpoint / resume, stop signals, seeding and logging.

May import: ``ovsb.config``, ``ovsb.data``, ``ovsb.model``, ``ovsb.losses``,
``wespeaker.utils.*`` and torch. Nothing below this layer imports it.
"""
