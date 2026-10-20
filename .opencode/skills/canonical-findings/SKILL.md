---
name: canonical-findings
description: Base canônica de conhecimento do repo (findings/) — consultar antes de responder/decidir com fatos do projeto, registrar todo conhecimento importante descoberto em novo arquivo + INDEX.md, e manter playbooks. Use quando o usuário mencionar "findings", "base canônica", "o que já se sabe sobre X", "registre isso", "atualize o INDEX", ou sempre que descobrir números/decisões/armadilhas estáveis durante qualquer tarefa.
---

# Base canônica findings/

`findings/` é a fonte determinística do que já se sabe sobre o projeto.
`agents/` = how-to operacional; `findings/` = conhecimento consolidado
(fatos, números, decisões, pendências); skills = runbooks de execução.

## Quando consultar (SEMPRE)

1. **Antes de responder perguntas factuais sobre o projeto** (arquitetura,
   protocolo, resultados, estado dos artigos): comece por
   `findings/INDEX.md` e leia os findings das áreas envolvidas. Não
   "adivinhe" fatos que podem estar lá.
2. Antes de citar QUALQUER número de resultado: finding
   `resultados-canonicos` + conferir o `evaluation.json` da rodada.
3. Ao trabalhar com relatórios/artigos/Optuna: ler o playbook
   correspondente em `findings/playbooks/` (que aponta para a skill
   autoritativa).

## Quando registrar (sempre que descobrir algo estável)

Conhecimento importante descoberto em qualquer tarefa → registrar:

```bash
python findings/scripts/findings.py new <slug> --title "..." --area <area>
```

- Já existe finding do tema? **ATUALIZE**-o (nunca duplique; atualize
  `updated:`).
- Preencher fatos verificáveis + seção **Fontes** (paths/comandos).
- Adicionar linha ao `INDEX.md` (`fix` adiciona faltantes com resumo
  "(preencher)").
- Receitas passo a passo recorrentes → `--playbook`.
- Contradição com finding antigo → corrigir o antigo na hora (a base não
  acumula verdades expiradas).

Antes de encerrar a sessão: `python findings/scripts/findings.py check`
(deve passar: tudo indexado, front-matter completo).

## Não vai para findings

- Detalhes operacionais duplicados das skills (aponte para a skill).
- Estado efêmero (jobs rodando, logs).
- Conteúdo de arquivos gerados (`reports/_site/`, `state.json`,
  `knowledge/index.md`, `references.bib`, `notebooks/colab_round2/`).
