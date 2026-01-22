class MSELoss:
    """Mean Squared Error for regression."""

    def __call__(self, predictions, targets):
        # predictions: iterable of Value (or numbers)
        # targets: iterable of Value (or numbers)
        predictions = list(predictions)
        targets = list(targets)

        n = len(predictions)
        if n == 0:
            raise ValueError("predictions must be non-empty")

        loss = sum((p - t) ** 2 for p, t in zip(predictions, targets))
        return loss * (1.0 / n)