import numpy as np
import pywt
from scipy import stats


def wavelet_denoising(signal, wavelet='sym18', level=2):
    """
    Aplica técnica de denoising wavelet para reduzir o ruído do sinal
    
    Parâmetros:
    signal: array-like - O sinal de entrada a ser processado
    wavelet: str - O tipo de wavelet a ser utilizado (default: 'sym18')
    level: int - O nível de decomposição wavelet (default: 2)
    
    Retorna:
    array-like - Sinal reconstruído após a remoção de ruído
    """
    coeffs = pywt.wavedec(signal, wavelet, mode='periodization', level=level)

    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]

    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode='periodization')
    return reconstructed_signal[:len(signal)]


def diebold_mariano_loss_diff(d, h=1, hac_lag=None, hln=True):
    """
    Núcleo do teste de Diebold–Mariano (1995) sobre a série de diferenciais
    de perda d_t = L(e1_t) - L(e2_t).

    H0: E[d_t] = 0 (precisão preditiva igual). A variância da média usa o
    estimador HAC para janelas sobrepostas (autocovariâncias de d até
    ``hac_lag``; default h−1). Com ``hln=True`` (default) aplica a correção
    de pequena amostra de Harvey, Leybourne e Newbold (1997) e avalia contra
    uma t de Student com n−1 graus de liberdade; com ``hln=False`` usa a
    normal assintótica (útil para a variante com banda automática de
    Newey–West — passe ``hac_lag=newey_west_lag(n)``).

    Parâmetros:
        d: array-like - diferenciais de perda (perda do modelo 1 menos a do 2)
        h: int - horizonte de previsão em passos (ordem da correlação de d)
        hac_lag: int - defasagem do HAC (None → h−1)
        hln: bool - aplica a correção de Harvey–Leybourne–Newbold

    Retorna:
        dict com n, dm (positiva = modelo 1 pior), p_value (bilateral),
        mean_loss_diff e hac_lag
    """
    lag = max(h - 1, 0) if hac_lag is None else int(hac_lag)
    return _dm_from_diff(d, hac_lag=lag, hln_h=(h if hln else 0))


def _hac_var_of_mean(centered, lag):
    """Variância HAC da média com truncamento em ``lag`` (Newey–West).

    gamma_0 + 2·Σ_{k=1..lag} gamma_k, tudo dividido por n — o estimador
    consistente para séries com autocorrelação até ``lag`` (janelas de
    previsão h-passos sobrepostas ⇒ lag = h−1).
    """
    n = centered.size
    total = float(np.dot(centered, centered) / n)
    for k in range(1, min(lag, n - 1) + 1):
        total += 2.0 * float(np.dot(centered[:-k], centered[k:]) / n)
    return total / n


def _dm_from_diff(d, hac_lag, hln_h=0):
    """Núcleo do DM: HAC com ``hac_lag`` + (opcional) correção HLN.

    ``hln_h`` = 0 desliga a correção de pequena amostra (usada pela variante
    com banda automática). Com ``hln_h >= 1`` aplica a correção de Harvey,
    Leybourne e Newbold (1997) e avalia contra t(n−1); caso contrário usa a
    normal assintótica do DM original.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    nan = {"n": int(n), "dm": np.nan, "p_value": np.nan,
           "mean_loss_diff": np.nan, "hac_lag": int(hac_lag)}
    if n <= 1:
        return nan
    dbar = float(d.mean())
    centered = d - dbar
    var_dbar = _hac_var_of_mean(centered, int(hac_lag))
    if not np.isfinite(var_dbar) or var_dbar <= 0:
        return {"n": int(n), "dm": np.nan, "p_value": np.nan,
                "mean_loss_diff": dbar, "hac_lag": int(hac_lag)}
    dm = dbar / np.sqrt(var_dbar)
    if hln_h and hln_h >= 1:
        correction = np.sqrt(
            (n + 1.0 - 2.0 * hln_h + hln_h * (hln_h - 1.0) / n) / n)
        dm = dm * correction
        p_value = float(2.0 * stats.t.sf(abs(dm), df=n - 1))
    else:
        p_value = float(2.0 * stats.norm.sf(abs(dm)))
    return {"n": int(n), "dm": float(dm), "p_value": p_value,
            "mean_loss_diff": dbar, "hac_lag": int(hac_lag)}


def newey_west_lag(n):
    """Banda automática de Newey–West (1994): floor(4·(n/100)^(2/9))."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def diebold_mariano(e1, e2, h=1, power=2):
    """
    Teste de Diebold–Mariano entre duas séries pareadas de erros de previsão.

    A perda de cada erro é |e|**power (power=2 → perda quadrática,
    consistente com o MSE). Consulte diebold_mariano_loss_diff para o resto
    da convenção (H0, correções e retorno).
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("e1 e e2 devem ter o mesmo formato")
    if e1.size == 0:
        return {"n": 0, "dm": np.nan, "p_value": np.nan,
                "mean_loss_diff": np.nan}
    return diebold_mariano_loss_diff(np.abs(e1) ** power - np.abs(e2) ** power,
                                     h=h)


def diebold_mariano_newey_west(e1, e2, power=2, lag=None):
    """DM com variância HAC e **banda automática de Newey–West (1994)**.

    Difere do DM clássico em dois pontos: (i) a defasagem do estimador HAC
    não é o horizonte h−1, mas a regra data-dependente
    ``floor(4·(n/100)^(2/9))``; (ii) a estatística é avaliada contra a
    normal assintótica N(0,1) (sem a correção de pequena amostra de Harvey).
    Use como checagem de robustez da escolha de banda: se o veredito mudar
    em relação ao DM(HLN+HAC), a autocorrelação residual é sensível ao lag.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    if e1.shape != e2.shape:
        raise ValueError("e1 e e2 devem ter o mesmo formato")
    d = np.abs(e1) ** power - np.abs(e2) ** power
    lag = newey_west_lag(d.size) if lag is None else int(lag)
    return _dm_from_diff(d, hac_lag=lag, hln_h=0)


def clark_west_stat(f, lag=None):
    """Estatística MSPE-ajustada de Clark–West a partir da série ``f_t``.

    Use quando já se tem a série de ajuste ``f_t`` (por período/origem);
    ``clark_west`` abaixo a constrói a partir das predições.
    """
    f = np.asarray(f, dtype=float)
    n = f.size
    fbar = float(f.mean())
    lag = newey_west_lag(n) if lag is None else int(lag)
    var = _hac_var_of_mean(f - fbar, lag)
    if not np.isfinite(var) or var <= 0:
        return {"n": int(n), "cw": np.nan, "p_value": np.nan,
                "mean_f": fbar, "hac_lag": int(lag)}
    cw = fbar / np.sqrt(var)
    return {"n": int(n), "cw": float(cw),
            "p_value": float(2.0 * stats.norm.sf(abs(cw))),
            "mean_f": fbar, "hac_lag": int(lag)}


def clark_west(actual, benchmark_pred, model_pred, lag=None):
    """Teste MSPE-ajustado de Clark & West (2007) para modelos **aninhados**.

    Quando o benchmark (modelo restrito, ex.: persistência) está aninhado no
    modelo avaliado (irrestrito), o DM subestima o ganho do modelo maior —
    a variância sob a nula se anula. Clark & West corrigem com

        f_t = e_bench² − e_model² + (bench − model)²

    e testam ``mean(f) / sqrt(HAC_var(f)/n)`` contra N(0,1). **CW > 0**
    favorece o modelo (irrestrito); **CW < 0** favorece o benchmark. A
    avaliação é assintótica e unilateral no texto original (aqui o p-valor é
    bilateral, para simetria com o DM).

    A banda ``lag`` do HAC vem da regra automática de Newey–West quando
    None. A série ``f_t`` pode ser agregada por origem (como no DM) para
    absorver a dependência das janelas sobrepostas.
    """
    actual = np.asarray(actual, dtype=float)
    b = np.asarray(benchmark_pred, dtype=float)
    m = np.asarray(model_pred, dtype=float)
    if not (actual.shape == b.shape == m.shape):
        raise ValueError("actual, benchmark_pred e model_pred devem ter o "
                         "mesmo formato")
    e_b = actual - b
    e_m = actual - m
    # (bench − model) = (actual − model) − (actual − bench) = e_m − e_b
    f = e_b ** 2 - e_m ** 2 + (e_m - e_b) ** 2
    return clark_west_stat(f, lag=lag)


def adjust_pvalues(pvalues, method="holm"):
    """Ajuste de multiplicidade dos p-valores (Holm 1979 / Bonferroni).

    Recebe um array (ou lista) de p-valores de uma família de comparações e
    devolve os p-valores ajustados, monotônicos e limitados a 1. ``method``
    ∈ {"holm", "bonferroni"}. Os p-valores com NaN são preservados.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return out
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    if method == "bonferroni":
        adj = np.minimum(p[order] * m, 1.0)
    elif method == "holm":
        ranked = p[order]
        factors = m - np.arange(m)
        adj = np.minimum(ranked * factors, 1.0)
        adj = np.maximum.accumulate(adj)
    else:
        raise ValueError("method deve ser 'holm' ou 'bonferroni'")
    out[order] = adj
    return out


def mase(actual, predicted, naive, power=1):
    """Mean Absolute Scaled Error — erro do modelo escalado pelo ingênuo.

    ``MASE = MAE_modelo / MAE_ingênuo`` (se ``power=1``): normaliza o erro do
    modelo pelo **erro intrínseco** da série (o da persistência). < 1 ⇒ o
    modelo bate o ingênuo; ≈ 1 ⇒ empate; > 1 ⇒ pior. Referência: Hyndman &
    Koehler (2006).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    naive = np.asarray(naive, dtype=float)
    denom = float(np.mean(np.abs(naive) ** power))
    if denom == 0:
        return np.nan
    return float(np.mean(np.abs(actual - predicted) ** power) / denom)


def _stationary_bootstrap_idx(n, mean_block, rng):
    """Índices do stationary bootstrap (Politis & Romano, 1994).

    Blocos de comprimento geométrico com média ``mean_block``; rápido
    (vetorizado via posições de reinício acumuladas).
    """
    p_restart = 1.0 / max(1.0, float(mean_block))
    starts = rng.integers(0, n, n)
    kick = rng.random(n) < p_restart
    kick[0] = True
    last = np.where(kick, np.arange(n), 0)
    last = np.maximum.accumulate(last)
    return (starts[last] + (np.arange(n) - last)) % n


def model_confidence_set(losses, names=None, alpha=0.10, n_boot=2000,
                         mean_block=36, seed=42):
    """Model Confidence Set (Hansen, Lunde & Nason, Econometrica 2011).

    Agrega a comparação sobre a predição inteira: ``losses`` é a matriz
    (T origens × M modelos) de perdas por origem — por exemplo, a média do
    erro quadrático sobre os 36 horizontes de cada origem, o que Remove a
    multiplicidade de testes pareados do Diebold–Mariano.

    Procedimento: para o conjunto ativo, estatística t_i da perda relativa
    à média do conjunto (com variância amostral), p-valores pela
    distribuição bootstrap do máximo das estatísticas (stationary bootstrap
    com blocos de média ``mean_block``, capturando a dependência de janelas
    sobrepostas); o pior modelo (menor p) é eliminado enquanto p < alpha.

    Retorna dict: {"mcs": [nomes no conjunto], "exclusion_p": {nome: p},
    "alpha", "n_origins", "names"} — p do excluído = p no round em que
    saiu; os remanescentes têm p >= alpha.
    """
    losses = np.asarray(losses, dtype=float)
    T, M = losses.shape
    names = list(names) if names is not None else [f"m{i}" for i in range(M)]
    rng = np.random.default_rng(seed)
    active = list(range(M))
    exclusion_p: dict[str, float] = {}

    while active:
        sub = losses[:, active]
        rel = sub - sub.mean(axis=1, keepdims=True)
        dbar = rel.mean(axis=0)
        var = rel.var(axis=0, ddof=1)
        se = np.sqrt(var / T)
        t_stat = np.divide(dbar, se, out=np.zeros_like(dbar), where=se > 0)

        boot_t = np.empty((n_boot, len(active)))
        for b in range(n_boot):
            idx = _stationary_bootstrap_idx(T, mean_block, rng)
            rel_b = sub[idx] - sub[idx].mean(axis=1, keepdims=True)
            db = rel_b.mean(axis=0) - dbar          # recentra sob H0
            var_b = rel_b.var(axis=0, ddof=1)
            se_b = np.sqrt(var_b / T)
            boot_t[b] = np.divide(db, se_b,
                                  out=np.zeros_like(db), where=se_b > 0)
        boot_max = boot_t.max(axis=1)
        pvals = np.array([(boot_max >= t).mean() for t in t_stat])

        worst = int(np.argmin(pvals))
        model_idx = active[worst]
        exclusion_p[names[model_idx]] = float(pvals[worst])
        if pvals[worst] >= alpha:
            break
        active.pop(worst)

    return {
        "mcs": [names[i] for i in active],
        "exclusion_p": exclusion_p,
        "alpha": float(alpha),
        "n_origins": int(T),
        "names": names,
        "mean_block": int(mean_block),
        "n_boot": int(n_boot),
    }
