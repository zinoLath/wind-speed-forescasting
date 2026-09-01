"""Shared hyperparameter search space for the TCN-based wrappers."""

TCN_RANGES = {
    "encoder": {"max_filters": 256, "max_dilation": 5},
    "decoder": {"max_filters": 128, "max_dilation": 4},
}


def tcn_hyperparameters(hp, side, filters, kernel_size, nb_stacks, dropout_rate, dilation_rate):
    """Collect the search space for one TCN block ("encoder" or "decoder").

    The keyword arguments are the historical defaults of each wrapper. The
    dilation list is derived from the sampled dilation_rate as powers of two,
    matching the original per-wrapper implementations.
    """
    ranges = TCN_RANGES[side]
    values = {
        "filters": hp.Int(f"{side}_filters", 32, ranges["max_filters"], step=16, default=filters),
        "kernel_size": hp.Int(f"{side}_kernel_size", 2, 4, default=kernel_size),
        "nb_stacks": hp.Int(f"{side}_nb_stacks", 1, 2, default=nb_stacks),
        "dropout_rate": hp.Float(
            f"{side}_dropout_rate", 0.0, 0.5, step=0.05, default=dropout_rate
        ),
        "dilation_rate": hp.Int(
            f"{side}_dilation_rate", 1, ranges["max_dilation"], default=dilation_rate
        ),
    }
    values["dilations"] = [2 ** i for i in range(values["dilation_rate"])]
    return values
