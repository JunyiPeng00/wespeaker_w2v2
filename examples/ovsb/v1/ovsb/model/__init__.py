"""Model: frontend subclass, classifier head, head record, builder.

Responsibility: build the speaker model (frontend, pooling backend,
classifier head) with the state-dict keys of the stock wespeaker model, and
record / verify which speaker list a classifier head belongs to.

May import: ``ovsb.config``, ``wespeaker.frontend``, ``wespeaker.models``,
``wespeaker.utils.frontend_utils`` and torch. Never imports ``ovsb.data``,
``ovsb.losses`` or ``ovsb.engine``.
"""
