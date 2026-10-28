"""Loading and merging of the pipeline configuration.

``pipeline.default.json`` holds the defaults and ``pipeline.json`` the explicit
user configuration. The user file is merged over the defaults, so it only needs
to declare the values the user actually wants to change.

Values shared by several stages (seed, window sizes, target column, loss, ...)
live in the ``common`` section and are inherited by every stage section whose
value is not explicitly set. Stage-level values always win over ``common``.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "pipeline.default.json"
USER_CONFIG_PATH = PIPELINE_DIR / "pipeline.json"

# Top-level sections that are not stages and therefore do not inherit from
# ``common``.
NON_STAGE_SECTIONS = ("paths", "gpu", "common")


def _deep_merge(base, override):
    """Recursively merge *override* into *base*; override always wins."""
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _inherit_common(config):
    """Fill each stage section with values from the shared ``common`` section.

    Resolution happens once at load time, so stages simply read their keys
    (``cfg["seed"]``, ...) without knowing where the value came from.
    """
    common = config.pop("common", {})
    for name, section in config.items():
        if name not in NON_STAGE_SECTIONS and isinstance(section, dict):
            config[name] = {**common, **section}
    return config


def load_config(path=None):
    """Load, merge and resolve the pipeline configuration.

    Precedence (later wins): ``pipeline.default.json`` →
    ``pipeline.json`` (canônico, versionado) → ``pipeline.local.json``
    (persistência local, gitignored — só aplicada sem ``--config``).

    Args:
        path: optional path to a user config file. When omitted,
            ``pipeline/pipeline.json`` is used if it exists. With an
            explicit path the local overlay is bypassed on purpose:
            perfis/rodadas Nome-dados têm precedência total e determinística.
    """
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)

    user_path = Path(path) if path else USER_CONFIG_PATH
    if user_path.is_file():
        with open(user_path, encoding="utf-8") as handle:
            config = _deep_merge(config, json.load(handle))
    elif path:
        # um --config explicito inexistente tem que falhar, nao cair no default
        raise FileNotFoundError(f"Config explicito nao encontrado: {user_path}")
    if path is None and LOCAL_CONFIG_PATH.is_file():
        with open(LOCAL_CONFIG_PATH, encoding="utf-8") as handle:
            config = _deep_merge(config, json.load(handle))
    return _inherit_common(config)


# ---------------------------------------------------------------------------
# Persistência de configuração (camada local sobre o canônico)
#
# A rodada canônica é sempre o default versionado (pipeline.json). Ajustes
# de experimento vivem em pipeline/pipeline.local.json (gitignored) e perfis
# reutilizáveis em pipeline/profiles/<nome>.json — o canônico nunca é
# editado por uma rodada.

LOCAL_CONFIG_PATH = PIPELINE_DIR / "pipeline.local.json"
PROFILES_DIR = PIPELINE_DIR / "profiles"


def _parse_value(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _split_key(key):
    section, dot, leaf = key.partition(".")
    if not dot or not section or not leaf:
        raise SystemExit(f"chave '{key}' inválida — use SECTION.KEY (ex.: train.epochs)")
    return section, leaf


def local_overrides():
    """Overrides ativos no pipeline.local.json (dict vazio se ausente)."""
    if not LOCAL_CONFIG_PATH.is_file():
        return {}
    with open(LOCAL_CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def set_values(pairs):
    """Grava SECTION.KEY=VALUE no arquivo local (cria se não existir)."""
    local = local_overrides()
    for key, raw in pairs:
        section, leaf = _split_key(key)
        local.setdefault(section, {})[leaf] = _parse_value(raw)
        print(f"local: {section}.{leaf} = {json.dumps(local[section][leaf])}")
    LOCAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(local, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def unset_values(keys):
    local = local_overrides()
    for key in keys:
        section, leaf = _split_key(key)
        if leaf in local.get(section, {}):
            del local[section][leaf]
            if not local[section]:
                del local[section]
            print(f"local: removido {section}.{leaf}")
        else:
            print(f"local: {section}.{leaf} não estava definido")
    if local:
        with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as handle:
            json.dump(local, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    elif LOCAL_CONFIG_PATH.exists():
        LOCAL_CONFIG_PATH.unlink()


def reset_local():
    if LOCAL_CONFIG_PATH.exists():
        LOCAL_CONFIG_PATH.unlink()
        print(f"removido {LOCAL_CONFIG_PATH.relative_to(PIPELINE_DIR.parent)} — "
              f"pipeline de volta ao canônico (pipeline.json)")
    else:
        print("nenhuma configuração local ativa")


def diff_local():
    """Diferença do local sobre o canônico (pipeline.json)."""
    with open(USER_CONFIG_PATH, encoding="utf-8") as handle:
        canon = json.load(handle)
    local = local_overrides()
    if not local:
        print("nenhuma configuração local ativa — pipeline no canônico")
        return
    for section, leaves in sorted(local.items()):
        for leaf, value in sorted(leaves.items()):
            base = canon.get(section, {}).get(leaf, "<ausente>")
            print(f"{section}.{leaf}: canônico={json.dumps(base)} → local={json.dumps(value)}")


def save_profile(name):
    if not local_overrides():
        raise SystemExit("nada para salvar — nenhuma configuração local ativa")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROFILES_DIR / f"{name}.json"
    with open(LOCAL_CONFIG_PATH, encoding="utf-8") as src, \
            open(dest, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(f"perfil salvo: {dest}")


def use_profile(name, keep_local=False):
    src = PROFILES_DIR / f"{name}.json"
    if not src.is_file():
        raise SystemExit(f"perfil inexistente: {src}")
    with open(src, encoding="utf-8") as handle:
        data = handle.read()
    if not keep_local and LOCAL_CONFIG_PATH.exists():
        LOCAL_CONFIG_PATH.unlink()
    with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as handle:
        handle.write(data)
    print(f"perfil '{name}' ativo em pipeline.local.json")


def list_profiles():
    if not PROFILES_DIR.is_dir():
        return
    profiles = sorted(PROFILES_DIR.glob("*.json"))
    if not profiles:
        return
    print("perfis:", ", ".join(p.stem for p in profiles))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Persistência de configuração da pipeline: ajustes locais "
                    "e perfis sobre o canônico (pipeline.json), sem editá-lo.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show", help="config resolvida (default+canônico+local)")
    p_set = sub.add_parser("set", help="grava SECTION.KEY=VALUE no local")
    p_set.add_argument("pairs", nargs="+", metavar="KEY=VALUE")
    p_unset = sub.add_parser("unset", help="remove SECTION.KEY do local")
    p_unset.add_argument("keys", nargs="+")
    sub.add_parser("diff", help="mostra o que o local sobrepõe ao canônico")
    sub.add_parser("reset", help="apaga o local e volta ao canônico")
    p_save = sub.add_parser("profile-save", help="salva o local como perfil")
    p_save.add_argument("name")
    p_use = sub.add_parser("profile-use", help="ativa um perfil salvo")
    p_use.add_argument("name")
    p_use.add_argument("--keep-local", action="store_true")
    sub.add_parser("profiles", help="lista perfis salvos")
    args = parser.parse_args()

    if args.cmd == "show":
        cfg = load_config()
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        if LOCAL_CONFIG_PATH.is_file():
            print(f"\n# overlay local ativo: {LOCAL_CONFIG_PATH}", file=sys.stderr)
    elif args.cmd == "set":
        pairs = []
        for item in args.pairs:
            key, eq, raw = item.partition("=")
            if not eq:
                raise SystemExit(f"esperado KEY=VALUE, recebido: {item}")
            pairs.append((key, raw))
        set_values(pairs)
    elif args.cmd == "unset":
        unset_values(args.keys)
    elif args.cmd == "diff":
        diff_local()
    elif args.cmd == "reset":
        reset_local()
    elif args.cmd == "profile-save":
        save_profile(args.name)
    elif args.cmd == "profile-use":
        use_profile(args.name, keep_local=args.keep_local)
    elif args.cmd == "profiles":
        list_profiles()


if __name__ == "__main__":
    main()