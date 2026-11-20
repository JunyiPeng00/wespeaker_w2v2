"""Losses: consistency, pairwise verification loss, loss bundle.

Responsibility: loss functions and their small running state. Every
function takes tensors (embeddings, labels, pair roles, arm indices), never
a batch dict or a config object.

May import: torch only. Never imports another ``ovsb`` layer.
"""
