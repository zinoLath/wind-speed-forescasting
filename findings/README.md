# findings/ — base canônica de conhecimento do repositório

Esta pasta é a **fonte canônica e determinística do que já se sabe** sobre o
projeto: código, dados, protocolos, resultados, artigos e processos. Os
arquivos em `agents/` são o *how-to operacional* por área; aqui vive o
**conhecimento consolidado** (fatos, números, decisões, pendências).

## Regras

1. **Conhecimento importante descoberto → novo arquivo em `knowledge/`**
   (ou atualização do arquivo existente do tema — nunca duplicar) **+ uma
   linha no `INDEX.md`**.
2. Todo finding tem front-matter (`title`, `area`, `updated`) e seção
   **Fontes** com paths/comandos verificáveis.
3. `INDEX.md` é o ponto de entrada: se um agente precisa de contexto,
   começa por ele. Entrada nova = link + resumo de 1 linha.
4. **Playbooks** (`playbooks/`) são receitas passo a passo de tarefas
   recorrentes (relatórios, retreino, integração no artigo, Overleaf, Optuna).
5. Antes de encerrar uma sessão que alterou findings, rode:

   ```bash
   python findings/scripts/findings.py check
   ```

## Scripts

```bash
python findings/scripts/findings.py list              # lista por área
python findings/scripts/findings.py search <termo>    # busca textual
python findings/scripts/findings.py show <slug>       # imprime um finding
python findings/scripts/findings.py new <slug> --title "..." --area protocolo
python findings/scripts/findings.py new <slug> --title "..." --playbook
python findings/scripts/findings.py check             # valida INDEX/front-matter
python findings/scripts/findings.py fix               # adiciona entradas faltantes
```

> Skill do agente: **canonical-findings**
> (`.opencode/skills/canonical-findings/SKILL.md`) — descreve quando
> consultar e como registrar findings. Os arquivos em `agents/` apontam
> para ela.
