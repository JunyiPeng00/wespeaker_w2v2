"""Training data: tar index, speaker table, episodic sampler, processors.

Responsibility: everything between the files on disk and a collated batch
(index, speaker and condition tables, sampling policies, cropping,
augmentation, channel simulation, collation, the data loader).

May import: ``ovsb.config``, ``wespeaker.dataset.*``, numpy and torch.
Never imports ``ovsb.model``, ``ovsb.losses`` or ``ovsb.engine``.
"""
