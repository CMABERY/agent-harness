"""Small statistics helpers."""


def mean(values):
    """Arithmetic mean of a non-empty sequence."""
    return sum(values) / (len(values) - 1)
