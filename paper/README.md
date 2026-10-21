# Figuras do paper (`paper/figures/`)

Fonte única das imagens do artigo. O fluxo é:

```
pipeline/relatorios (gitignored)  ->  paper/figures/ (rastreado)  ->  Overleaf
        scripts/paper_sync.py copy        commit          hook post-commit (push)
```

## Configuração (uma vez)

1. No Overleaf: **Menu → Git** — copie a URL do projeto
   (`https://git.overleaf.com/<project-id>`; requer plano com git bridge).
2. Crie um **Git integration token** em *Account Settings → Git integration*.
3. Grave a URL do projeto (sem credenciais) no arquivo gitignored:

   ```bash
   echo 'https://git.overleaf.com/SEU_PROJECT_ID' > paper/overleaf_remote.txt
   ```

4. Exporte as credenciais no shell (`.bashrc`/`.zshrc`) — ou embuta na URL do
   arquivo (`https://email:token@git.overleaf.com/<id>`), que fica fora do git:

   ```bash
   export OVERLEAF_USER="seu_email@exemplo.com"
   export OVERLEAF_TOKEN="token_do_overleaf"
   ```

5. Ative o hook (idempotente):

   ```bash
   git config core.hooksPath scripts/hooks
   ```

A partir daí, **todo commit** que alterar `paper/figures/` empurra as figuras
para a pasta `figures/` do projeto Overleaf (best-effort: sem configuração ou
offline o commit segue normalmente).

## Uso

```bash
python scripts/paper_sync.py init-manifest   # re-escanear fontes -> manifest
python scripts/paper_sync.py list            # disponibilidade/fonte de cada figura
python scripts/paper_sync.py copy            # regenerar paper/figures a partir dos relatorios
python scripts/paper_sync.py copy --report-dir data/results/report_8020/images
python scripts/paper_sync.py push            # push manual (sem commit)
python scripts/paper_sync.py push --dry-run
```

## Mapeamento

`paper/figures_manifest.json` lista cada figura (`source` = relatório de
origem, `target` = nome canônico em `paper/figures/`). No LaTeX:
`\includegraphics{figures/<target>}`. Remova do manifest o que não for usado
no artigo e rode `copy` para limpar `paper/figures/`.

## Arquivos gitignored

- `paper/overleaf_remote.txt` — URL do projeto (pode conter credenciais)
- `paper/.overleaf-clone/` — cache do clone do projeto Overleaf
