---
title: Registrar um finding (fluxo canônico)
area: playbook
updated: 2026-10-19
---

# Playbook: registrar um finding

Sempre que descobrir conhecimento importante e estável (números, decisões,
armadilhas, estados de artigo/rodada):

1. Verificar se já existe finding do tema (`python findings/scripts/findings.py
   list` ou `search <termo>`):
   - **Existe** → ATUALIZAR o arquivo (conteúdo + `updated:` na data de
     hoje). Não criar duplicata;
   - **Não existe** → criar:
     ```bash
     python findings/scripts/findings.py new <slug-curto> \
         --title "Título curto" --area <area>
     ```
     Áreas: infra, arquitetura, dados, protocolo, resultados, metodos,
     relatorios, artigos, processo. (Receita passo a passo → `--playbook`.)
2. Preencher conteúdo com fatos verificáveis + seção **Fontes** (paths,
   comandos, artefatos);
3. Adicionar a linha no `INDEX.md` (ou usar `fix`):
   `- [slug](knowledge/slug.md) — resumo de 1 linha`;
4. Validar:
   ```bash
   python findings/scripts/findings.py check
   ```
5. Se o conhecimento contradiz um finding antigo: atualizar o antigo na
   hora (ou marcar a revisão nele) — a base não deve acumular verdades
   expiradas.

O que NÃO vai para findings: detalhes operacionais duplicados das skills
(aponte para a skill), estado efêmero (jobs rodando), conteúdo de
`reports/_site/` (gerado).
