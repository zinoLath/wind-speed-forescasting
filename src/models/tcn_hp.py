"""Shared hyperparameter search space for the TCN-based wrappers.

Reduced after the round-2 studies: a single shared ``filters`` value feeds
both encoder and decoder, the kernel stays small (2-3), the in-block dropout
caps at 0.4 and the dilation upper bounds shrink (the ``min_receptive_field``
extension still guarantees the sampled stack covers the forecast window).
"""

TCN_RANGES = {
    "filters": {"max_filters": 192},
    "encoder": {"max_dilation": 4},
    "decoder": {"max_dilation": 3},
}


def tcn_receptive_field(kernel_size, nb_stacks, dilations):
    """Receptive field of a TCN stack (matches the ``tcn`` library)."""
    return 1 + 2 * (kernel_size - 1) * nb_stacks * sum(dilations)


def tcn_hyperparameters(
    hp, side, filters, kernel_size, nb_stacks, dropout_rate, dilation_rate,
    min_receptive_field=0, filters_value=None,
    static_dilations=None, depth_range=None,
):
    """Collect the search space for one TCN block ("encoder" or "decoder").

    The keyword arguments are the historical defaults of each wrapper. The
    dilation list is derived from the sampled dilation_rate as powers of two,
    matching the original per-wrapper implementations. When
    *filters_value* is given (e.g. a shared ``filters`` hp sampled once by
    the wrapper) it is used verbatim for both blocks.

    When *min_receptive_field* is given (e.g. the input/output window), the
    dilation list is extended with larger powers of two until the receptive
    field covers it, so a sampled config can never see less history than the
    forecast window. The extension is deterministic given the sampled
    hyperparameters, so replaying a saved best_trial.json reproduces the same
    model.

    Modo estático (round 3, opt-in): quando *static_dilations* é dado, a
    sequência de dilatações é FIXA (ex.: [1,2,4,8,16,32,36]) e o único HP de
    dilatação é a profundidade ``<side>_dil_depth`` (prefixo do array,
    limitado por *depth_range*). Nesse modo a extensão automática de campo
    receptivo é DESLIGADA (enxergar menos história é uma configuração
    válida), e kernel_size/nb_stacks ficam fixos nos argumentos passados.
    """
    if static_dilations:
        lo, hi = depth_range or (len(static_dilations), len(static_dilations))
        depth = hp.Int(f"{side}_dil_depth", min_value=int(lo),
                       max_value=int(hi), default=int(hi))
        if filters_value is not None:
            filters_value = filters_value
        else:
            filters_value = hp.Int(
                f"{side}_filters", 32, TCN_RANGES["filters"]["max_filters"],
                step=16, default=filters,
            )
        return {
            "filters": filters_value,
            "kernel_size": hp.Int(f"{side}_kernel_size", 2, 3,
                                  default=kernel_size),
            "nb_stacks": hp.Int(f"{side}_nb_stacks", 1, 2,
                                default=nb_stacks),
            "dropout_rate": hp.Float(
                f"{side}_dropout_rate", 0.0, 0.4, step=0.05,
                default=dropout_rate,
            ),
            "dia_depth": depth,
            "dilations": list(static_dilations[:depth]),
        }

    if filters_value is not None:
        values = {"filters": filters_value}
    else:
        values = {"filters": hp.Int(
            f"{side}_filters", 32, TCN_RANGES["filters"]["max_filters"],
            step=16, default=filters,
        )}
    values.update({
        "kernel_size": hp.Int(f"{side}_kernel_size", 2, 3, default=kernel_size),
        "nb_stacks": hp.Int(f"{side}_nb_stacks", 1, 2, default=nb_stacks),
        "dropout_rate": hp.Float(
            f"{side}_dropout_rate", 0.0, 0.4, step=0.05, default=dropout_rate
        ),
        "dilation_rate": hp.Int(
            f"{side}_dilation_rate", 1, TCN_RANGES[side]["max_dilation"],
            default=dilation_rate
        ),
    })
    dilations = [2 ** i for i in range(values["dilation_rate"])]
    while min_receptive_field and tcn_receptive_field(
        values["kernel_size"], values["nb_stacks"], dilations
    ) < min_receptive_field:
        dilations.append(2 ** len(dilations))
    values["dilations"] = dilations
    return values

# Sequências estáticas de dilatação (convenção C′ do round 4): encoder com o
# lag sazonal de 36 passos (6 h) direto; decoder truncado ao máximo útil.
STATIC_DILATIONS = {
    "encoder": [1, 2, 4, 8, 16, 32, 36],
    "decoder": [1, 2, 4, 8, 16],
    "encoder_depth": (4, 7),
    "decoder_depth": (3, 5),
}


def apply_static_dilations(wrapper, hyperparameters):
    """Liga o modo estático de dilatações quando os hp indicam (round 4).

    Modelos treinados no modo estático gravam chaves ``*_dil_depth`` nos
    hiperparâmetros, mas NÃO as sequências. Na remontagem (relatórios/
    avaliação) é preciso restaurar as sequências também — sem isso a
    arquitetura difere (modo legado) e o ``load_weights`` falha.
    """
    if any(k.endswith("_dil_depth") for k in (hyperparameters or {})):
        wrapper.static_dilations = STATIC_DILATIONS
    return wrapper
