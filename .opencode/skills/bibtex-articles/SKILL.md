---
name: bibtex-articles
description: Consulta e completa o catálogo BibTeX da coleção de artigos (articles/knowledge/) — lista títulos para o usuário escolher, devolve a entrada bibtex COM abstract, lê os downloads (PDF/extração markdown) quando faltam metadados. Use quando o usuário pedir "bibtex de um artigo", "achar o abstract", "citação do paper X", "atualizar references.bib", ou mencionar knowledge/references.bib.
---

# Catálogo BibTeX dos artigos (knowledge)

Fontes: `articles/knowledge/references.bib` (GERADO — não editar à mão),
`articles/knowledge/index.md` (índice gerado, aponta PDF/texto/página de
cada artigo) e os downloads em `articles/knowledge/pdf/` +
`articles/knowledge/markdown/`. Gerador: `articles/knowledge/tools/knowledge.py`.

## 1. Mostrar a lista de títulos (sempre comece por aqui)

```bash
.venv/bin/python articles/knowledge/tools/knowledge.py list      # tudo (chave + título)
.venv/bin/python articles/knowledge/tools/knowledge.py search lstm   # filtra título/abstract/autores
```

Apresente ao usuário uma lista numerada curta (título + ano + chave) para
ele escolher. Com a chave em mãos:

```bash
.venv/bin/python articles/knowledge/tools/knowledge.py bibtex <chave-ou-termo>  # imprime a entrada .bib
.venv/bin/python articles/knowledge/tools/knowledge.py show <chave-ou-termo>    # metadados + caminhos
```

## 2. Entrada sem abstract (marcada ⚠ no índice) ou metadado errado

1. `show <chave>` informa os caminhos: `[PDF](pdf/...)`, `[texto](markdown/...)`,
   `[página](pages/...)`.
2. Leia o **texto extraído** (`markdown/*.md`) — os abstracts ficam após
   `A B S T R A C T`/`## Abstract`; extraia título, autores, ano, veículo e
   DOI do cabeçalho do arquivo.
3. Se só existe o PDF, extraia o texto da 1ª página antes de ler:

```bash
.venv/bin/python - <<'EOF'
import pymupdf
doc = pymupdf.open("articles/knowledge/pdf/<arquivo>.pdf")
print(doc[0].get_text()[:3500])
EOF
```

4. Grave os metadados no lugar certo e regenere (nunca edite o .bib gerado):
   - opção preferida: crie/corrija a extração em
     `articles/knowledge/markdown/<mesmo-stem-do-pdf>.md`;
   - alternativa: ajuste o parser/heurística em `tools/knowledge.py` se o
     formato do publisher não for coberto;
   - depois rode `build` e confira no índice que o ⚠ sumiu:

```bash
.venv/bin/python articles/knowledge/tools/knowledge.py build
```

## 3. Artigo que ainda não está no catálogo

- PDF novo da máquina → salve em `~/documents/papers/` (o symlink
  `articles/knowledge/pdf/documents` enxerga automaticamente) ou copie
  para `pdf/` (terceiros) / `pdf/own-work/` (trabalhos próprios; adicione
  o título manual em `OWN_WORK_TITLES` no knowledge.py se o metadado do PDF
  vier errado) e rode `build`.
- Sem PDF → adicione a extração `.md` (com abstract) em
  `articles/knowledge/markdown/` e rode `build`.

## 4. Usar no Overleaf/paper

- A bib dos projetos LaTeX é separada (`articles/sbr/LaTeX_Template_SBSR_Portugues/07_SBSR.bib`,
  `articles/steam/referencias.bib`): copie a entrada gerada para lá quando o
  usuário pedir, mantendo o abstract como campo `abstract = {...}` (ignorado
  pelo `\bibliography` padrão, útil para revisão).
- Chaves seguem `sobrenomeANOprimeirapalavra` (ex.: `venkateswaran2026hybrid`).
