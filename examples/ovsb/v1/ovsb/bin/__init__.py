"""Command-line entry points.

Responsibility: thin argument parsing around the other layers (config
validation, training, dumps). No logic that a test would need to import.

May import: any ``ovsb`` layer. ``validate_config`` restricts itself to
``ovsb.config`` so that it runs without torch.
"""
