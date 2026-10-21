# Sessão de Comentários SBSR #1

Sessão de 20/10/2026. Consolidação mecânica do estado da revisão do
artigo SBSR (`articles/sbr/LaTeX_Template_SBSR_Portugues/`), a partir de:

- `docs/revisions/sbr/comentarios_revisao_sbsr.md` — análise comentário
  a comentário da rodada Overleaf (13–14/10/2026; 18 threads, Vanessa +
  Flávio) + pendências herdadas;
- `docs/comentarios_vanessa.md` — rodada do commit `151ac29` (typos e
  citações);
- comentários inseridos nos `.tex` nesta sessão (blocos
  `% COMENTÁRIOS DE REVISÃO` em `02`, `03`, `04`, `06`).

**Convenções:** IDs (`6ace…`) = threads do export Overleaf. `[V-…]` =
Vanessa, `[F-…]` = Flávio. Este relatório **não contém prosa de artigo** —
roteiros de conteúdo (para redigir) vivem no doc de revisão; aqui estão estado,
números, comandos e material de estudo.

Panorama (estado pós-push Overleaf, 20/10): **3 a fazer · 2 pendentes de
execução (aguardam o retreino) · 4 com decisão humana · 9 resolvidos**.
Esforços são estimativas de tempo humano (salvo indicação de máquina).

## Kanban

<div class="kbsbsr" id="kb1">
<style>
.kbsbsr{margin:14px 0}
.kb-toolbar{display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.kb-hint{color:#667;font-size:12.5px}
.kb-reset{cursor:pointer;border:1px solid #d5dde6;background:#fff;border-radius:6px;padding:3px 10px;font-size:12.5px}
.kb-board{display:flex;gap:12px;align-items:flex-start;overflow-x:auto;padding-bottom:6px}
.kb-col{flex:1 1 0;min-width:210px;background:#f2f6fa;border:1px solid #d5dde6;border-radius:10px;padding:8px}
.kb-col h3{margin:2px 4px 8px;font-size:13.5px;display:flex;justify-content:space-between;align-items:center;gap:6px}
.kb-count{background:var(--accent2,#1c6fb8);color:#fff;border-radius:10px;padding:0 8px;font-size:11.5px}
.kb-total{font-weight:400;color:#667;font-size:11.5px}
.kb-body{display:flex;flex-direction:column;gap:8px;min-height:36px}
.kb-card{background:#fff;border:1px solid #d5dde6;border-left:4px solid #2b7bb9;border-radius:8px;padding:8px 10px;font-size:13px;cursor:grab;box-shadow:0 1px 2px rgba(20,40,60,.06)}
.kb-card:active{cursor:grabbing}
.kb-card.kb-drag{opacity:.45}
.kb-col[data-col="exec"] .kb-card{border-left-color:#c77d0a}
.kb-col[data-col="human"] .kb-card{border-left-color:#8a5fbf}
.kb-col[data-col="done"] .kb-card{border-left-color:#3c9e57}
.kb-col[data-col="done"] .kb-card .kb-title{text-decoration:line-through;color:#556}
.kb-id{color:#889;font-size:11px;font-weight:400}
.kb-eff{display:inline-block;margin-top:6px;background:#eef4fa;border:1px solid #d5dde6;color:#33526b;border-radius:10px;padding:1px 8px;font-size:11.5px}
</style>
<div class="kb-toolbar">
  <span class="kb-hint">Arraste os cartões entre colunas (reordenação local, salva no navegador — o texto abaixo é a fonte canônica).</span>
  <button type="button" class="kb-reset">restaurar layout</button>
</div>
<div class="kb-board">
  <div class="kb-col" data-col="todo">
    <h3>A fazer (determinístico) <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-13" data-min="60"><div class="kb-title">§1.3 Tabelas de hiperparâmetros e do dataset <span class="kb-id">V-6ace8ce9</span></div><span class="kb-eff">≈1 h (gerar de JSONs)</span></div>
      <div class="kb-card" draggable="true" data-id="c-14" data-min="30"><div class="kb-title">§1.4 Contagem 7632 × 7561 (contradição "sem valores ausentes") <span class="kb-id">V-6ace7f52</span></div><span class="kb-eff">≈30 min</span></div>
      <div class="kb-card" draggable="true" data-id="c-15" data-min="60"><div class="kb-title">§1.5 Figuras legíveis (P1 na fig. 1, fonte do heatmap) <span class="kb-id">V-6ace71fd / 992b</span></div><span class="kb-eff">≈1 h (regenerar + resync)</span></div>
    </div>
  </div>
  <div class="kb-col" data-col="exec">
    <h3>Pendente execução/redação (pós-re treino) <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-11" data-min="25"><div class="kb-title">§1.1 Retreino com HPs round5 — EXECUTADO E REVERTIDO (transferência negativa; canônica = 16/10 restaurada; HPs round5 = rodada própria `8020-round5`) <span class="kb-id">V-6ace944b</span></div><span class="kb-eff">✓ executado (21/10) — decisão: restaurar 16/10</span></div>
      <div class="kb-card" draggable="true" data-id="c-12" data-min="60"><div class="kb-title">§1.2 DM no artigo: roteiro + números prontos (docs/revisions/sbr/roteiro_dm.md) — redigir <span class="kb-id">V-6ace9381</span></div><span class="kb-eff">≈1 h (redação sua)</span></div>
    </div>
  </div>
  <div class="kb-col" data-col="human">
    <h3>Aguardando decisão humana <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-22" data-min="15"><div class="kb-title">§2.2 Reescrever parágrafo da loss ("tá correto?") — texto no 03 + evidência em `protocolo/itildeq-termos` <span class="kb-id">V-6ace8d78</span></div><span class="kb-eff">≈15 min</span></div>
      <div class="kb-card" draggable="true" data-id="c-24" data-min="15"><div class="kb-title">§2.4 Completar bibliotecas/versões (só "Python" foi citado) <span class="kb-id">V-6ace820a</span></div><span class="kb-eff">≈15 min (colar)</span></div>
      <div class="kb-card" draggable="true" data-id="c-28" data-min="30"><div class="kb-title">§2.8 Aplicar reescritas das Conclusões (recapitular objetivo, P1, CV temporal) <span class="kb-id">F-6ace2e79/2f14/2fa9</span></div><span class="kb-eff">≈30 min</span></div>
      <div class="kb-card" draggable="true" data-id="c-29" data-min="30"><div class="kb-title">§2.9 Entradas bib dos baselines (Box & Jenkins, Breiman, Hyndman) <span class="kb-id">Apêndice D</span></div><span class="kb-eff">≈30 min (verificar refs)</span></div>
    </div>
  </div>
  <div class="kb-col" data-col="done">
    <h3>Feito <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-d1"><div class="kb-title">Typos de 02/03/06 corrigidos</div><span class="kb-eff">✓</span></div>
      <div class="kb-card" draggable="true" data-id="c-d2"><div class="kb-title">Tabela fora das referências (placeins + FloatBarrier)</div><span class="kb-eff">✓</span></div>
      <div class="kb-card" draggable="true" data-id="c-d3" data-min="45"><div class="kb-title">ANEEL atualizado (SIGA, 35 GW, 2ª fonte, cite aneelSiga) <span class="kb-id">V-6ace4915</span></div><span class="kb-eff">✓ (sessão 20/10)</span></div>
      <div class="kb-card" draggable="true" data-id="c-d4"><div class="kb-title">arcilanVentoSul (Sea breeze 2024) × EOSOLAR (projeto 2022) esclarecidos <span class="kb-id">V-6ace641f</span></div><span class="kb-eff">✓ (Overleaf)</span></div>
      <div class="kb-card" draggable="true" data-id="c-d5"><div class="kb-title">§2.1 Rodada canônica ESCOLHIDA: mista 8020-itildeq (cadeia + vigia ajustados)</div><span class="kb-eff">✓ (sessão 20/10)</span></div>
      <div class="kb-card" draggable="true" data-id="c-d6"><div class="kb-title">§2.5 Verbos pessoais — Introdução reescrita impessoal <span class="kb-id">F-6ace2c7f</span></div><span class="kb-eff">✓ (Overleaf)</span></div>
      <div class="kb-card" draggable="true" data-id="c-d7"><div class="kb-title">§2.3 Entendimento do negócio: contexto down-ramps/Vento Sul <span class="kb-id">V-6ace64b6</span></div><span class="kb-eff">✓ (Overleaf)</span></div>
      <div class="kb-card" draggable="true" data-id="c-d8"><div class="kb-title">§2.6 Terminologia "arquiteturas" adotada <span class="kb-id">F-6ace2e9e</span></div><span class="kb-eff">✓ (Overleaf)</span></div>
      <div class="kb-card" draggable="true" data-id="c-d9"><div class="kb-title">§2.7 Legenda "Fonte:" na fig. 1 <span class="kb-id">V-6ace641f</span></div><span class="kb-eff">✓ (Overleaf)</span></div>
    </div>
  </div>
</div>
<script>
(function(){
  var root=document.getElementById('kb1'); if(!root) return;
  var KEY='kanban.sbsr1.v1', board=root.querySelector('.kb-board');
  function defLayout(){return [].map.call(board.querySelectorAll('.kb-col'),function(c){return {col:c.dataset.col,ids:[].map.call(c.querySelectorAll('.kb-card'),function(k){return k.dataset.id;})};});}
  var DEF=defLayout();
  function cardMap(){var m={};[].forEach.call(board.querySelectorAll('.kb-card'),function(k){m[k.dataset.id]=k;});return m;}
  function apply(layout){var m=cardMap();layout.forEach(function(col){var el=board.querySelector('.kb-col[data-col="'+col.col+'"]');if(!el)return;var body=el.querySelector('.kb-body');(col.ids||[]).forEach(function(id){if(m[id])body.appendChild(m[id]);});});}
  function counts(){[].forEach.call(board.querySelectorAll('.kb-col'),function(c){var ks=c.querySelectorAll('.kb-card'),min=0;[].forEach.call(ks,function(k){min+=parseInt(k.getAttribute('data-min')||'0',10);});var cnt=c.querySelector('.kb-count');if(cnt)cnt.textContent=ks.length;var tot=c.querySelector('.kb-total');if(tot)tot.textContent=min?('\u2248'+(min>=60?(Math.round(min/6)/10).toString().replace('.',',')+' h':min+' min')+' de esfor\u00e7o'):'';});}
  function save(){try{localStorage.setItem(KEY,JSON.stringify(defLayout()));}catch(e){}counts();}
  function restore(){var s=null;try{s=JSON.parse(localStorage.getItem(KEY)||'null');}catch(e){}apply(s&&s.length?s:DEF);counts();}
  var dragEl=null;
  board.addEventListener('dragstart',function(e){var c=e.target.closest('.kb-card');if(!c)return;dragEl=c;setTimeout(function(){c.classList.add('kb-drag');},0);e.dataTransfer.setData('text/plain',c.dataset.id);e.dataTransfer.effectAllowed='move';});
  board.addEventListener('dragend',function(e){if(dragEl)dragEl.classList.remove('kb-drag');dragEl=null;save();});
  board.addEventListener('dragover',function(e){if(!dragEl)return;e.preventDefault();var col=e.target.closest('.kb-col');if(!col)return;var body=col.querySelector('.kb-body');var next=null;[].some.call(body.querySelectorAll('.kb-card'),function(k){if(k===dragEl)return false;var r=k.getBoundingClientRect();if(e.clientY<r.top+r.height/2){next=k;return true;}return false;});if(next)body.insertBefore(dragEl,next);else if(dragEl.parentElement!==body)body.appendChild(dragEl);});
  board.addEventListener('drop',function(e){e.preventDefault();});
  root.querySelector('.kb-reset').addEventListener('click',function(){try{localStorage.removeItem(KEY);}catch(e){}apply(DEF);counts();});
  restore();
})();
</script>
</div>

> Fluxo: o retreino da canônica (coluna 2) desbloqueia §1.2 e dá os
> números para as tabelas/texto; detalhes de cada cartão nas seções
> indicadas. Cartões marcados "(Overleaf)" foram atendidos pelos
> coautores na edição remota de 20/10 (commit `a6ea1cf`).

---

## 1. Comentários em aberto (trabalho determinístico pendente)

Ordem por impacto. Nenhum item aqui exige opinião — são passos mecânicos
com critério de pronto verificável.

### 1.1 [RESOLVIDA A ESCOLHA — PENDENTE EXECUÇÃO] Retreino da rodada canônica — `V-6ace944b`

> **Atualização 20/10/2026:** a rodada canônica foi **ESCOLHIDA**: a
> **mista `8020-itildeq`** (TCN/TCN-Bi = MSE + HPs canônicos + dilatações
> estáticas C′; LSTM/LSTM-Bi = ITILDE-Q + HPs do HPO). O **treino será
> executado depois**; toda a cadeia já está ajustada para gerar os dados
> corretos quando rodar.

- **Cadeia canônica (ajustada e pronta):**
  1. Backup: `cp -r pipeline/tmp/trained_8020_itildeq pipeline/tmp/trained_8020_itildeq_backup_$(date +%Y%m%d%H%M)`;
  2. Treino: `python scripts/lidar_8020_itildeq_train.py` (losses/HPs por modelo embutidos → `pipeline/tmp/trained_8020_itildeq`);
  3. Regeneração **SEM `--no-baselines`**: `python tests/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_itildeq --out-dir data/results/report_8020_itildeq` — DL e baselines (ARIMA/RF/persistência) saem na **mesma grade de origens** (o alinhamento está em `tests/relatorio_resultados.py` ~linha 360);
  4. Critério de pronto: `MSE/(1−R²)` **igual para as 7 linhas** (var implícita única — fecha a inconsistência da Tabela 1);
  5. Tabelas LaTeX: `python scripts/sbsr_tables.py --round-dir data/results/report_8020_itildeq`;
  6. Figuras do artigo: `python scripts/articles_sync.py sync sbr` — **`sync.json` já aponta para `report_8020_itildeq`**;
  7. Integração no texto (tabelas + Resumo + loss): skill `sbsr-integrate`;
  8. Rebuild do hub: `python reports/src/builder.py build`.
- Runbook autoritativo atualizado: skill `lidar-retrain` + playbook
  `findings/playbooks/retreino-lidar.md`; decisão registrada no finding
  `resultados-canonicos`.
- **Sessão armada (20/10, pid em `pipeline/tmp/watch_retreino_itildeq.log`):**
  `scripts/watch_retreino_itildeq.sh` espera o treino começar e terminar
  (até 240 h) e, sozinho, calcula os pontos em um resultado novo —
  regeneração com baselines na mesma grade → `mcs.json` (DM) → tabelas
  `sbsr_tables` → rebuild do hub — e notifica (ntfy + desktop). Se o
  treino rodar no Kaggle, o vigia local não enxerga o kernel: traga os
  modelos para `trained_8020_itildeq` e rode a cadeia do passo 3
  manualmente. Sobrevive ao logout (`setsid`), não a reboot — rearmar com
  `setsid bash scripts/watch_retreino_itildeq.sh 240 < /dev/null >> pipeline/tmp/watch_retreino_itildeq.log 2>&1 & disown`.
- **Evidência histórica (por que era crítico):** pela eq. (1) do artigo,
  `Var(y) implícita = MSE/(1−R²)`: baselines da Tabela 1 → **2,702**
  (1478 origens = 53208 amostras) vs DL → **2,779** (≈1406 origens =
  50616). Origens diferentes ⇒ R² não comparáveis; e o heatmap publicado
  (rodada ponderada) contradizia a ordenação da tabela.

### 1.2 [MÉTODO RESOLVIDO — PENDENTE EXECUÇÃO] Teste DM no artigo — `V-6ace9381` + `V-6ace9991`

> **Atualização 20/10/2026:** metodologia **resolvida** — mas precisa dos
> resultados do retreino (§1.1) para ser executada e os p-valores irem
> ao artigo. Status mantido como **pendente execução** no kanban.

- **Estado (reverificado 20/10):** ferramenta existe e **evoluiu** —
  além de `tests/relatorio_resultados.py` (`diebold_mariano_loss_diff`)
  → `mcs.json` (14/10), o relatório **`reports/lidar/8020-itildeq-metodos/`**
  traz DM **ajustado** (Harvey–Leybourne–Newbold + HAC/Newey–West
  automático, correção de Holm, Clark–West) — exatamente o upgrade
  sugerido pela Vanessa. Falta **usar** no artigo.
- **Passos:**
  1. Rodar o comando do item 1.1 (gera DM junto);
  2. Ler pares/p-valores em `data/results/report_8020/mcs.json`;
  3. Inserir: 1 parágrafo no §Avaliação (definição H₀ + por que
     HAC/blocos — parágrafo sugerido no doc de revisão §3.8/[METODO-A1]
     do `03`) + tabela de p-valores dos pares TCN×LSTM,
     TCN-Bi×LSTM-Bi, melhor DL×ARIMA;
  4. Corrigir os números do par. 3 do Resultados: as diferenças de R²
     são **0,080** e **0,042** (o texto diz "0,8" e "0,4") — pendente
     também `V-6ace9991`.
- **Estudo:** §4.2 deste relatório.

### 1.3 Tabelas de hiperparâmetros e do dataset — `V-6ace8ce9`

- **Dados exatos (verificados):**
  - hiperparâmetros: `data/results/optuna/<modelo>/trials.csv` (best em
    `data/results/wfo_optuna/<modelo>/metrics.json`);
  - nº de parâmetros / loss / tamanho: `evaluation.json` por modelo;
  - tempo de treino: `data/results/*/summary.json`
    (`training_time_sec`), `training_history.json`;
  - dataset: colunas de `data/dataset.csv` — coletados `ws{40..260}` (20
    níveis), `dir{...}`, `v{...}`, `press`, `humid`, `temp`;
    computados `cis1..cis19`, `disp{...}`/`vdisp{...}` (confirmar
    dicionário exato).
- **Passo:** montar as duas tabelas (posso gerar dos JSONs — pedir).
- **Decisão embutida:** listar só os 4 modelos canônicos e citar
  ablações em texto (ver doc de revisão §3.6).

### 1.4 Contagem de amostras 7632 × 7561 — `V-6ace7f52`

- **Verificado:** `data/dataset.csv` = 7561 linhas, 16/09 18:00 →
  08/11/2021; série **não completa** (linhas inteiras ausentes, sem
  NaN); grade completa do intervalo ≈ 7606–7776 conforme a contagem.
- **Pendente (mecânico):** confirmar se `dataset.csv` é pós-imputação
  (estágio `impute`, `src/impute_wind_data.py`, lacunas ≤ 12 passos = 2 h)
  ou bruto da campanha — comparar contra o CSV bruto da campanha
  (`data/wind_data.csv` cobre período diferente; localizar o export da
  campanha EOSOLAR). Depois: declarar no texto N ausentes + tratamento
  (roteiro de conteúdo no doc de revisão §3.4).

### 1.5 Figuras legíveis — `V-6ace71fd` (P1) + `V-6ace992b` (fonte)

> **Reverificado em 20/10/2026:** nenhuma das duas mudou — fig. 1 ainda
> a `0.5\linewidth`; heatmap-fonte (`06_heatmap_mse_horizonte.png`,
> 14/10) sem regeneração, fontsize dos rótulos ainda ~8–8,5 em
> `tests/relatorio_resultados.py`.

- **Heatmap:** regenerar `06_heatmap_mse_horizonte.png` com fontsize
  ~13–16 (ou figsize menor) em `tests/relatorio_resultados.py`;
  re-sincronizar com `python scripts/articles_sync.py sync`.
- **Figura 1 (P1):** escolher abordagem (ampliar para `\linewidth` /
  recorte com P1 centralizado / regerar com marcador destacado);
  conferir se a figura do Overleaf difere da local. Regerar imagem é
  mecânico; a escolha da abordagem está no §2.

---

## 2. Atuados — falta intervenção humana (escrever / dissertar / escolher)

Respostas já verificadas com dados/código; o que falta é decisão ou
redação sua. Textos prontos no doc de revisão, seções indicadas.

| # | Item (thread) | O que já está feito | O que falta de você |
|---|---|---|---|
| 1 | ~~Escolha da rodada canônica~~ **RESOLVIDA (20/10)** | Decidida a **mista `8020-itildeq`** (TCN 0,519/TCN-Bi 0,490 com MSE; LSTM 0,318/Bi 0,329 com ITILDE-Q — números da execução 16/10; serão atualizados pelo retreino). Cadeia ajustada (§1.1) | — (só executar o retreino quando rodar) |
| 2 | "Tá correto?" — loss ( `V-6ace8d78`) | Pesos 0,7→1,3 corretos (`src/models/losses.py:15-25`); com a canônica escolhida, o texto deve dizer "LSTM ITILDE-Q / TCN MSE" — roteiro de conteúdo no bloco [METODO-LOSS] do `03` | Redigir com os números do retreino |
| 3 | Entendimento do negócio (`V-6ace64b6`) | Diagnóstico: subseção descreve dados, não negócio; esqueleto de 4 pontos pronto (doc §3.1) | Dissertar o negócio (geração eólica, Delta Maranhão, por que prever importa) |
| 4 | Ferramentas/versões (`V-6ace820a`) | Versões verificadas (Python 3.10, TF 2.21, PyWavelets 1.8 com algoritmos prontos, sklearn 1.7, statsmodels 0.15, Optuna 4.9); parágrafo pronto (doc §3.5) | Revisar e colar |
| 5 | Verbos pessoais (`F-6ace2c7f`, resolvido no Overleaf mas texto mantém) | Reescrita impessoal pronta (doc §1.1) | Aplicar (ou decidir manter) |
| 6 | "Arquitetura" (`F-6ace2e9e`) | Regra de troca definida (arquitetura = combinação estrutura+rede; rede neural = mecanismo) | Varrer o texto e aplicar o critério |
| 7 | Legenda "Fonte:" + EOSOLAR (`V-6ace641f`) | Bib local `arcilanVentoSul` **já é** o artigo do projeto EOSOLAR (Assireu et al. 2022, Energies 15(7):2319) | Conferir o bib no **Overleaf**; escolher formato da legenda (doc §3.3) |
| 8 | Conclusões: recapitular objetivo / "área de estudo" / validação cruzada (`F-6ace2e79`, `F-6ace2f14`, `F-6ace2fa9`) | 3 reescritas prontas (doc §5.1–5.3); diagnóstico correto: dado é só do P1; CV temporal existe (Roberts 2016, já no bib) | Revisar e aplicar |
| 9 | Citações de baselines (Apêndice D) | Referências identificadas: Box & Jenkins (ARIMA), Breiman 2001 (RF), Hyndman & Athanasopoulos (persistência) | Criar entradas com dados verificados (skill `bibtex-articles`) |

---

## 3. Resolvidos (simples)

| Item | Thread | Resolução |
|---|---|---|
| Typos das Conclusões (Contanto, limtação, codificador/codificador, se demonstrou) | rodada `151ac29` | Corrigidos nos `.tex` locais (doc §5.4) |
| Typos de Material e métodos (tem, foi encontrada, implementados) | rodada `151ac29` | Corrigidos (doc §5.4) |
| Tabela renderizando nas referências | `F-6ace2fd7` | `\usepackage{placeins}` + `\FloatBarrier` no preâmbulo |
| **ANEEL desatualizado** | `V-6ace4915` | **Esta sessão** — ver tabela abaixo; frase reescrita + `@misc{aneelSiga}` no bib |
| Float `[h]` / Placeins validado | — | Pendente só validação visual no PDF do Overleaf |
| `arcilanVentoSul` = projeto EOSOLAR | `V-6ace641f` (parte) | Verificado no bib **local** (resta conferir Overleaf — item 7 do §2) |

### ANEEL — números desta sessão (SIGA, snapshot 2026-10-20)

Fonte: `data/siga-empreendimentos-geracao-diario.csv`
(dadosabertos.aneel.gov.br; autor SCE/ANEEL; mantenedor CEGDI/ANEEL;
versão 1.1; atualização 20/10/2026; 25.058 empreendimentos). Filtro:
`DscFaseUsina = "Operação"`, potência **outorgada** (MdaPotenciaOutorgadaKw).

| Posição | Fonte | GW | Participação |
|---|---|---|---|
| 1º | Hídrica | 110,3 | 49,8% |
| **2º** | **Eólica** | **35,0** | **15,8%** |
| 3º | Fóssil | 32,1 | 14,5% |
| 4º | Solar | 23,8 | 10,8% |
| 5º | Biomassa | 18,3 | 8,3% |
| 6º | Nuclear | 2,0 | 0,9% |
| — | Total | 221,6 | 100% |

Detalhes eólica: 1.138 empreendimentos em operação; UFs: BA 381, RN 319,
PI 126, CE 103, RS 80, PE 46; **MA 16 emp. / 0,4 GW**; pipeline futuro:
2,0 GW em construção (62) + 14,2 GW não iniciada (348). Matriz
renovável (E+S+H+B): 84,7% (fiscalizada). Fiscalizada eólica: 34,9 GW.

Correção conceitual que a troca resolve: o texto antigo dizia "22 GW na
**geração**, terceiro lugar" — misturava capacidade com geração e fonte
errada (EOSOLAR não é fonte de matriz elétrica). Agora: capacidade
(SIGA/ANEEL) com cite próprio; o cite EOSOLAR ficou só para o ranking
global de 2023.

---

## 4. O que estudar (para resolver o que falta — em ordem)

Cada item: por quê → conceito mínimo → material exato → exercício
determinístico de fixação.

### 4.1 R² × variância implícita (30 min — destrava 1.1)

- **Por quê:** é a chave do [CRÍTICO]; explica por que a Tabela 1 não
  pode estar certa.
- **Conceito:** pela própria eq. (1) do artigo,
  `R² = 1 − MSE/Var(y)` ⇒ `Var(y) = MSE/(1−R²)`. Se todos os modelos
  foram avaliados sobre os mesmos valores reais, **todas as linhas
  produzem a mesma variância implícita**. Valores diferentes ⇒ conjuntos
  de avaliação diferentes.
- **Exercício:** abrir `data/results/report_8020/evaluation.json` e
  `data/results/baselines/arima_rf/metrics.json`; calcular MSE/(1−R²)
  por linha e comparar. Repetir com a Tabela 1 do artigo e constatar os
  grupos 2,70 × 2,78.
- **Material:** doc de revisão §4.2 (conta completa feita).

### 4.2 Diebold–Mariano + bootstrap em blocos (o pedido explícito da Vanessa)

- **Por quê:** threads `6ace9381`/`6ace9991`; sem isso, "TCN > LSTM" é
  alegação sem significância.
- **Conceito mínimo:**
  - por origem do rolling forecast, `d_t = L(e_A,t) − L(e_B,t)`
    (perda, ex. MSE). H₀: `E[d_t] = 0`;
  - a estatística DM normaliza a média de `d_t` por um desvio-padrão
    estimado com autocovariâncias (**HAC/Newey–West**) porque `d_t` é
    serialmente correlacionado;
  - correção de amostras pequenas: **Harvey–Leybourne–Newbold (1997)**;
  - **bootstrap em blocos (moving-block):** reamostrar blocos contíguos
    de `d_t` (preserva a dependência temporal) para o p-valor/IC sem
    assumir normalidade.
- **Onde já está:** `tests/relatorio_resultados.py`
  (`diebold_mariano_loss_diff`, agregado e por horizonte) →
  `data/results/report_8020/mcs.json`.
- **Material:** doc de revisão Apêndice C; precedentes da sua coleção:
  `docs/revisions/sbr/kbscout/result-1.md` (gomez2026predicting: Harvey +
  Newey–West; bendes2026bio: Friedman→Wilcoxon+Holm) e `result-4.md`
  (Giacomini–White condicional).
- **Exercício:** rodar o comando do 1.1 e ler `mcs.json`: localizar o
  par TCN×LSTM e interpretar estatística + p-valor.

### 4.3 Métricas relativas: skill score e Theil's U (complemento barato)

- **Por quê:** formaliza "bateu o naive?" por horizonte; você já tem
  `skill_por_horizonte.csv`.
- **Conceito:**
  - **Skill score:** `SS = 1 − MSE_modelo/MSE_naive` (0 = empata com o
    naive; <0 = pior);
  - **Theil U2 (1966):** `RMSE_modelo/RMSE_naive = √(1−SS)` — U2 < 1 ⇔
    SS > 0. Mesma informação, escala diferente;
  - **U1/TIC** (o que parte da literatura de vento chama "Theil
    coefficient") é outro animal: índice em [0,1] sem benchmark —
    descritivo, não diz contra o quê você é bom;
  - Nenhuma das duas **testa significância** — complementam o DM, não
    substituem.
- **Material:** precedentes na coleção — `lin2022deep` (Applied Energy
  2022, cita Theil U como métrica consagrada em vento) e `chen2024research`
  (MSFSC, usa TIC + acurácia direcional); consolidado em
  `docs/revisions/sbr/kbscout/result-5.md`.
- **Exercício:** de `skill_por_horizonte.csv`, calcular U2 = √(1−SS) por
  horizonte e verificar que a curva cruza 1,0 onde a skill cruza 0.

### 4.4 Testes globais e por regime (camada seguinte ao DM)

- **Friedman** (não-paramétrico, múltiplos modelos × múltiplos blocos —
  ex.: 36 horizontes como blocos) → se significativo, **Wilcoxon
  pareado + correção de Holm** nos pares. Precedente:
  `bendes2026bio` (result-1).
- **Giacomini–White**: habilidade preditiva **condicional** (por regime
  de vento, ex. ≥13,9 m/s vs <) — o DM é incondicional. Precedente:
  `jie2026designing` (result-4). Encaixa com seus estratos
  regime/banda em `metodos.json`.
- **MCS (Model Confidence Set)**: você **já tem** (`mcs.json`) e
  **nenhum** artigo da coleção usa — diferencial se reportado.
- **Material:** `docs/revisions/sbr/kbscout/result-1.md` a `result-4.md`.

### 4.5 Validação cruzada temporal (contexto da limitação)

- **Por quê:** thread `F-6ace2fa9` — "não foi possível" está errado;
  existe CV com estrutura temporal.
- **Conceito:** k-fold clássico embaralha e viola a ordem temporal;
  *time series split / blocked CV / rolling-origin* respeita (treino em
  blocos anteriores, teste em posteriores disjuntos). Seu 80/20 é uma
  única dobra temporal; a limitação é não repetir em múltiplas dobras.
- **Material:** Roberts et al. 2016 (chave `crossvalidation`, **já no
  bib**); doc de revisão §5.3.

### 4.6 CRISP-DM: negócio ≠ dados (para o item 3 do §2)

- **Conceito:** "Entendimento do negócio" = contexto/objetivos sob a
  ótica do negócio (o que está em jogo: comercialização, despacho,
  O&M); "Entendimento dos dados" = instrumentação, período, atributos.
  A subseção atual tem o conteúdo da segunda.
- **Material:** `wirth2000crisp` (já no bib); esqueleto pronto no doc
  §3.1.

---

## 5. Ordem de execução sugerida (determinística)

1. Estudar §4.1 e §4.2 (1–2 h) → decidir a rodada canônica (item 1 do §2);
2. Item 1.1: recalcular tudo numa rodada única + conferir variância
   implícita única (critério objetivo de pronto);
3. Item 1.2: DM no artigo (parágrafo no §Avaliação + p-valores +
   corrigir "0,8"/"0,4");
4. Itens 1.3–1.5 (tabelas, contagem, figuras) — posso gerar as tabelas
   e as contagens quando pedido;
5. Decisões de prosa do §2 (uma sessão de escrita sua, com os textos
   prontos do doc de revisão à mão);
6. Antes de submeter: remover todos os blocos `% COMENTÁRIOS DE
   REVISÃO` dos `.tex` + `consideracoes_extras.tex`, e validar o PDF no
   Overleaf.
