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

> **Atualização 21/10:** o **LSTM-Bi está rodando em Optuna** (HPO para
> contornar a ITILDE-Q fraca — LSTM 0,318/LSTM-Bi 0,329 vs TCN 0,519 na
> canônica 16/10). Enquanto não fechar, **Resultados/Discussão e a
> seção de hiperparâmetros podem mudar de valor** — itens dependentes
> de números ficam como **P2** (executar só depois do treino). O que é
> independente de números (**P1**) pode ser zerado agora. Plano completo
> com esforço × importância × prioridade na **§6**. Contagem de amostras
> (§1.4) **fechada**: não é duplicata (zero timestamps duplicados; 50
> amostras ausentes na grade do próprio intervalo). Fix do
> `sbsr_tables.py` (§1.6) **já aplicado**.

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
.kb-pri{display:inline-block;margin-top:6px;margin-left:4px;border-radius:10px;padding:1px 8px;font-size:11.5px;border:1px solid transparent}
.kb-pri.p1{background:#fdeeee;border-color:#e4b6b6;color:#9c3030}
.kb-pri.p2{background:#fdf3e7;border-color:#e4cfae;color:#8a5a14}
.kb-pri.p3{background:#eef1f4;border-color:#ccd4dc;color:#51606e}
.kb-title a{color:inherit;text-decoration:none}
.kb-title a:hover{text-decoration:underline}
.kb-card .kb-eip{display:block;margin-top:6px;color:#667;font-size:11px}
.kb-filters{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 10px;background:#fff;border:1px solid #d5dde6;border-radius:8px;padding:6px 10px;font-size:12.5px}
.kb-filters label{display:flex;gap:5px;align-items:center;color:#33526b}
.kb-filters select{border:1px solid #d5dde6;border-radius:6px;padding:2px 6px;font-size:12.5px;background:#fff;color:#22364a}
.kb-clear{cursor:pointer;border:1px solid #d5dde6;background:#fff;border-radius:6px;padding:3px 10px;font-size:12.5px}
.kb-total-all{font-weight:600;color:#1c4f7c;background:#eef4fa;border:1px solid #d5dde6;border-radius:8px;padding:2px 10px}
h2,h3,h4{scroll-margin-top:12px}
a.anch{text-decoration:none}
</style>
<div class="kb-toolbar">
  <span class="kb-hint">Arraste os cartões entre colunas (reordenação local, salva no navegador — o texto abaixo é a fonte canônica).</span>
  <button type="button" class="kb-reset">restaurar layout</button>
</div>
<div class="kb-filters">
  <span class="kb-total-all" id="kb-total-all"></span>
  <label>tempo
    <select id="kb-f-tempo">
      <option value="all">todos</option>
      <option value="le30">≤ 30 min</option>
      <option value="m30_60">30–60 min</option>
      <option value="gt60">&gt; 1 h</option>
    </select></label>
  <label>prioridade
    <select id="kb-f-pri">
      <option value="all">todas</option>
      <option value="1">P1 (agora)</option>
      <option value="2">P2 (pós-treino)</option>
      <option value="3">P3 (fechamento)</option>
    </select></label>
  <label>importância
    <select id="kb-f-imp">
      <option value="all">todas</option>
      <option value="critica">crítica</option>
      <option value="alta">alta</option>
      <option value="media">média</option>
    </select></label>
  <button type="button" class="kb-clear">limpar filtros</button>
</div>
<div class="kb-board">
  <div class="kb-col" data-col="todo">
    <h3>A fazer (determinístico) <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-14" data-p="1" data-i="alta" data-min="30"><div class="kb-title"><a href="#sec-1-4">§1.4 Contagem 7632 × 7561 — ANÁLISE FECHADA (21/10: não é duplicata; 50 ausentes na grade); falta redigir no 03 <span class="kb-id">V-6ace7f52</span></a></div><span class="kb-eff">≈30 min (redigir)</span><span class="kb-pri p1" title="Importância: alta —corrige afirmação errada de série completa">P1 · I alta</span></div>
      <div class="kb-card" draggable="true" data-id="c-15" data-p="2" data-i="alta" data-min="60"><div class="kb-title"><a href="#sec-1-5">§1.5 Figuras legíveis — GERADOR JÁ AJUSTADO (21/10: fontsizes 19–21 nos heatmaps; fig. 1 a \linewidth); falta regenerar na rodada final + resync <span class="kb-id">V-6ace71fd / 992b</span></a></div><span class="kb-eff">≈1 h (regen + resync, pós-treino)</span><span class="kb-pri p2" title="Importância: alta — código pronto; execução pendente da rodada">P2 · I alta</span></div>
      <div class="kb-card" draggable="true" data-id="c-24" data-p="1" data-i="media" data-min="15"><div class="kb-title"><a href="#sec-2">§2.4 Completar bibliotecas/versões (parágrafo pronto no doc §3.5 — revisar e colar) <span class="kb-id">V-6ace820a</span></a></div><span class="kb-eff">≈15 min (colar)</span><span class="kb-pri p1" title="Importância: média">P1 · I média</span></div>
      <div class="kb-card" draggable="true" data-id="c-28" data-p="1" data-i="alta" data-min="30"><div class="kb-title"><a href="#sec-2">§2.8 Aplicar reescritas das Conclusões (recapitular objetivo, P1, CV temporal — 3 textos prontos, doc §5.1–5.3) <span class="kb-id">F-6ace2e79/2f14/2fa9</span></a></div><span class="kb-eff">≈30 min</span><span class="kb-pri p1" title="Importância: alta — 3 threads de coautor">P1 · I alta</span></div>
      <div class="kb-card" draggable="true" data-id="c-29" data-p="1" data-i="media" data-min="45"><div class="kb-title"><a href="#sec-2">§2.9 Entradas bib dos baselines (Box &amp; Jenkins, Breiman, Hyndman) + \cite{vaswani2017attention} nos Transformers das Conclusões — GRU NÃO será abordada (decisão 21/10) <span class="kb-id">Apêndice D</span></a></div><span class="kb-eff">≈45 min (verificar refs)</span><span class="kb-pri p1" title="Importância: média">P1 · I média</span></div>
      <div class="kb-card" draggable="true" data-id="c-13" data-p="2" data-i="alta" data-min="60"><div class="kb-title"><a href="#sec-1-3">§1.3 Tabelas de hiperparâmetros e do dataset — linhas de TCN podem ser geradas já; linhas de LSTM/LSTM-Bi esperam o HPO (números podem mudar) <span class="kb-id">V-6ace8ce9</span></a></div><span class="kb-eff">≈1 h (gerar de JSONs)</span><span class="kb-pri p2" title="Importância: alta — mas dependente do HPO do LSTM-Bi">P2 · I alta</span></div>
      <div class="kb-card" draggable="true" data-id="c-31" data-p="3" data-i="alta" data-min="30"><div class="kb-title"><a href="#sec-6">Pré-submissão: remover blocos <code>% COMENTÁRIOS DE REVISÃO</code> dos .tex + <code>consideracoes_extras.tex</code> <span class="kb-id">checklist</span></a></div><span class="kb-eff">≈30 min</span><span class="kb-pri p3" title="Importância: alta — bloqueia a submissão; executar por último">P3 · I alta</span></div>
      <div class="kb-card" draggable="true" data-id="c-32" data-p="3" data-i="alta" data-min="30"><div class="kb-title"><a href="#sec-6">Pré-submissão: validação final do PDF no Overleaf (floats, bib, figura = tabela da mesma rodada, var. implícita única) <span class="kb-id">checklist</span></a></div><span class="kb-eff">≈30 min</span><span class="kb-pri p3" title="Importância: alta — fecha o [CRÍTICO]">P3 · I alta</span></div>
    </div>
  </div>
  <div class="kb-col" data-col="exec">
    <h3>Pendente execução/redação (pós-re treino) <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-11" data-p="2" data-i="critica" data-min="60"><div class="kb-title"><a href="#sec-1-1">§1.1 Rodada final: EXECUTADO E REVERTIDO (transferência negativa; canônica = 16/10 restaurada; HPs round5 = rodada própria `8020-round5`) — agora AGUARDA o HPO do LSTM-Bi (Optuna em andamento, 21/10) para rodar a mista com os HPs novos <span class="kb-id">V-6ace944b</span></a></div><span class="kb-eff">≈1 h humano + ~30 min máquina</span><span class="kb-pri p2" title="Importância: crítica — fecha o [CRÍTICO]; bloqueado pelo HPO">P2 · I crítica</span></div>
      <div class="kb-card" draggable="true" data-id="c-12" data-p="2" data-i="critica" data-min="60"><div class="kb-title"><a href="#sec-1-2">§1.2 DM no artigo: roteiro + números prontos (docs/revisions/sbr/roteiro_dm.md) — redigir com os p-valores da rodada final <span class="kb-id">V-6ace9381</span></a></div><span class="kb-eff">≈1 h (redação sua)</span><span class="kb-pri p2" title="Importância: crítica — pedido explícito da Vanessa">P2 · I crítica</span></div>
      <div class="kb-card" draggable="true" data-id="c-22" data-p="2" data-i="alta" data-min="15"><div class="kb-title"><a href="#sec-2">§2.2 Reescrever parágrafo da loss ("tá correto?") — estrutura no 03 + evidência em `protocolo/itildeq-termos`; redigir com os números da rodada final ("LSTM ITILDE-Q / TCN MSE") <span class="kb-id">V-6ace8d78</span></a></div><span class="kb-eff">≈15 min</span><span class="kb-pri p2" title="Importância: alta — descrição da loss precisa bater com a rodada">P2 · I alta</span></div>
    </div>
  </div>
  <div class="kb-col" data-col="human">
    <h3>Aguardando decisão humana <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-33" data-p="2" data-i="media" data-min="15"><div class="kb-title"><a href="#sec-2">§2.3 Validar se o contexto de negócio vindo do Overleaf (commit `a6ea1cf`: down-ramps/Vento Sul) cobre o pedido da Vanessa, ou se falta dissertar (esqueleto pronto no doc §3.1) <span class="kb-id">V-6ace64b6</span></a></div><span class="kb-eff">≈15 min (ler + decidir)</span><span class="kb-pri p2" title="Importância: média">P2 · I média</span></div>
      <div class="kb-card" draggable="true" data-id="c-34" data-p="2" data-i="alta" data-min="30"><div class="kb-title"><a href="#sec-1-3-1">§2.9b Dicionário do dataset: frase de "Preparação dos dados" diz que entraram dispersão/cisalhamento, mas os modelos NÃO usam disp/vdisp/cis (§1.3.1, Tabela B) — ajustar o texto ou re-incluir os atributos; tabela canônica do dataset pronta na §1.3.1 <span class="kb-id">V-6ace8ce9</span></a></div><span class="kb-eff">≈30 min (decidir + redigir)</span><span class="kb-pri p2" title="Importância: alta — consistência do que se declara vs o que roda">P2 · I alta</span></div>
    </div>
  </div>
  <div class="kb-col" data-col="done">
    <h3>Feito <span class="kb-count">0</span><span class="kb-total"></span></h3>
    <div class="kb-body">
      <div class="kb-card" draggable="true" data-id="c-30" data-min="10"><div class="kb-title">§1.6 Fix `sbsr_tables.py` (ValueError `relative_to` no cabeçalho) — erro do log do vigia era pré-fix; `.resolve()` já aplicado e validado (rc=0)</div><span class="kb-eff">✓ (21/10)</span></div>
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
  var FT=document.getElementById('kb-f-tempo'),FP=document.getElementById('kb-f-pri'),FI=document.getElementById('kb-f-imp');
  function fmtH(min){if(!min)return'0 min';var h=Math.floor(min/60),r=min%60;return (h?h+' h':'')+(h&&r?' + ':'')+(r?r+' min':'');}
  function applyFilters(){
    var t=FT.value,p=FP.value,i=FI.value,n=0,min=0;
    [].forEach.call(board.querySelectorAll('.kb-card'),function(k){
      var m=parseInt(k.getAttribute('data-min')||'0',10);
      var okT=(t==='all')||(t==='le30'&&m<=30)||(t==='m30_60'&&m>30&&m<=60)||(t==='gt60'&&m>60);
      var okP=(p==='all')||k.getAttribute('data-p')===p;
      var okI=(i==='all')||k.getAttribute('data-i')===i;
      var vis=okT&&okP&&okI; k.style.display=vis?'':'none';
      var col=k.closest('.kb-col');
      if(vis&&col&&col.getAttribute('data-col')!=='done'){n++;min+=m;}
    });
    var tot=document.getElementById('kb-total-all');
    if(tot)tot.textContent='Restante: \u2248'+fmtH(min)+' ('+n+' cart\u00f5es em aberto, sem "Feito")';
  }
  [FT,FP,FI].forEach(function(el){el.addEventListener('change',applyFilters);});
  root.querySelector('.kb-clear').addEventListener('click',function(){FT.value='all';FP.value='all';FI.value='all';applyFilters();});
  restore();
  applyFilters();
})();
</script>
</div>

> Fluxo: o **HPO do LSTM-Bi (Optuna)** desbloqueia a rodada final (coluna
> 2), que desbloqueia §1.2 e dá os números para as tabelas/texto; os
> cartões P1 da coluna 1 são independentes de números e podem ser
> zerados agora. Os títulos dos cartões **linkam para a seção** deste
> relatório que descreve a demanda (âncoras `#sec-*`). Badges:
> **P1** = fazer agora (independe do treino) · **P2** = após a rodada
> final · **P3** = fechamento pré-submissão; **I** = importância
> (crítica/alta/média). Detalhes de cada cartão nas seções indicadas.
> Cartões marcados "(Overleaf)" foram atendidos pelos coautores na
> edição remota de 20/10 (commit `a6ea1cf`). Se o layout local do
> navegador estiver desatualizado, usar **restaurar layout**.

---

## 1. Comentários em aberto (trabalho determinístico pendente)

Ordem por impacto. Nenhum item aqui exige opinião — são passos mecânicos
com critério de pronto verificável.

<a id="sec-1-1"></a>
### 1.1 [RESOLVIDA A ESCOLHA — PENDENTE EXECUÇÃO] Retreino da rodada canônica — `V-6ace944b`

> **Atualização 20/10/2026:** a rodada canônica foi **ESCOLHIDA**: a
> **mista `8020-itildeq`** (TCN/TCN-Bi = MSE + HPs canônicos + dilatações
> estáticas C′; LSTM/LSTM-Bi = ITILDE-Q + HPs do HPO). O **treino será
> executado depois**; toda a cadeia já está ajustada para gerar os dados
> corretos quando rodar.
>
> **Atualização 21/10/2026:** treino com HPs round5 **executado e
> REVERTIDO** (transferência negativa nos TCNs; canônica 16/10
> restaurada). **Novo bloqueio/plano:** LSTM-Bi está rodando em
> **Optuna** para tentar resgatar a ITILDE-Q (LSTM 0,318/LSTM-Bi 0,329
> vs TCN 0,519). A rodada final só deve ser executada **quando esse HPO
> fechar**, com os HPs novos de LSTM/LSTM-Bi — enquanto isso, os números
> de Resultados/Discussão e da tabela de hiperparâmetros são
> provisórios (ver §6).

- **Cadeia canônica (ajustada e pronta):**
  1. Backup: `cp -r pipeline/tmp/trained_8020_itildeq pipeline/tmp/trained_8020_itildeq_backup_$(date +%Y%m%d%H%M)`;
  2. Treino: `python scripts/experiments/lidar_8020_itildeq_train.py` (losses/HPs por modelo embutidos → `pipeline/tmp/trained_8020_itildeq`);
  3. Regeneração **SEM `--no-baselines`**: `python reports/src/relatorio_resultados.py --models-dir pipeline/tmp/trained_8020_itildeq --out-dir data/results/report_8020_itildeq` — DL e baselines (ARIMA/RF/persistência) saem na **mesma grade de origens** (o alinhamento está em `reports/src/relatorio_resultados.py` ~linha 360);
  4. Critério de pronto: `MSE/(1−R²)` **igual para as 7 linhas** (var implícita única — fecha a inconsistência da Tabela 1);
  5. Tabelas LaTeX: `python scripts/sbsr_tables.py --round-dir data/results/report_8020_itildeq`;
  6. Figuras do artigo: `python scripts/articles_sync.py sync sbr` — **`sync.json` já aponta para `report_8020_itildeq`**;
  7. Integração no texto (tabelas + Resumo + loss): skill `sbsr-integrate`;
  8. Rebuild do hub: `python reports/src/builder.py build`.
- Runbook autoritativo atualizado: skill `lidar-retrain` + playbook
  `findings/playbooks/retreino-lidar.md`; decisão registrada no finding
  `resultados-canonicos`.
- **Sessão armada (20/10, pid em `pipeline/tmp/watch_retreino_itildeq.log`):**
  `scripts/dist/watch_retreino_itildeq.sh` espera o treino começar e terminar
  (até 240 h) e, sozinho, calcula os pontos em um resultado novo —
  regeneração com baselines na mesma grade → `mcs.json` (DM) → tabelas
  `sbsr_tables` → rebuild do hub — e notifica (ntfy + desktop). Se o
  treino rodar no Kaggle, o vigia local não enxerga o kernel: traga os
  modelos para `trained_8020_itildeq` e rode a cadeia do passo 3
  manualmente. Sobrevive ao logout (`setsid`), não a reboot — rearmar com
  `setsid bash scripts/dist/watch_retreino_itildeq.sh 240 < /dev/null >> pipeline/tmp/watch_retreino_itildeq.log 2>&1 & disown`.
- **Evidência histórica (por que era crítico):** pela eq. (1) do artigo,
  `Var(y) implícita = MSE/(1−R²)`: baselines da Tabela 1 → **2,702**
  (1478 origens = 53208 amostras) vs DL → **2,779** (≈1406 origens =
  50616). Origens diferentes ⇒ R² não comparáveis; e o heatmap publicado
  (rodada ponderada) contradizia a ordenação da tabela.

<a id="sec-1-2"></a>
### 1.2 [MÉTODO RESOLVIDO — PENDENTE EXECUÇÃO] Teste DM no artigo — `V-6ace9381` + `V-6ace9991`

> **Atualização 20/10/2026:** metodologia **resolvida** — mas precisa dos
> resultados do retreino (§1.1) para ser executada e os p-valores irem
> ao artigo. Status mantido como **pendente execução** no kanban.

- **Estado (reverificado 20/10):** ferramenta existe e **evoluiu** —
  além de `reports/src/relatorio_resultados.py` (`diebold_mariano_loss_diff`)
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

<a id="sec-1-3"></a>
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
- **Dependência 21/10:** as linhas de **LSTM/LSTM-Bi esperam o HPO**
  (Optuna em andamento) — gerar agora pode exigir regeneração. As linhas
  de TCN/TCN-Bi (HPs canônicos) e a tabela do dataset são estáveis e
  podem ser preparadas já.

<a id="sec-1-3-1"></a>
### 1.3.1 Dicionário de atributos — o que de fato entrou nos modelos (21/10)

Verificação direta de `data/dataset.csv` (128 colunas) passando por
`common.load_dataset` (`forecast_redundant_columns`): **o CSV bruto tem
muito mais do que os modelos consomem**. Entrada efetiva dos wrappers =
**46 canais + alvo**.

**Tabela A — ENTRA nos modelos (46 entradas + alvo):**

| Atributo | Qtd | Origem | Significado |
|---|---|---|---|
| `ws{40..140}` | 11 | **Coletado** (LiDAR) | Velocidade horizontal do vento (m/s) na altura h (m); alturas 40–140, passo de 10 m |
| `v{40..140}` | 11 | **Coletado** (LiDAR) | Componente vertical do vento (m/s) na altura h |
| `dir{h}_sin`, `dir{h}_cos` | 22 | **Computado** (de `dir{h}` coletado em graus) | Direção do vento projetada no círculo unitário (evita a descontinuidade 359°→1°) |
| `hour_sin`, `hour_cos` | 2 | **Computado** | Hora do dia codificada ciclicamente (periodicidade diária) |
| `ws100_wavelet` (alvo) | 1 | **Computado** (de `ws100`) | ws100 suavizada por wavelet discreta sym18, nível 2, limiar soft de Donoho–Johnstone |

**Tabela B — coletado/computado no CSV mas DESCARTADO antes dos modelos
(`forecast_redundant_columns`):**

| Atributo | Qtd | Origem | Motivo do descarte |
|---|---|---|---|
| `ws/v/dir{150..260}` | 27 | Coletado (LiDAR) | Alturas fora do conjunto canônico (40–140 m) |
| `disp{40..260}` | 20 | Computado | Dispersão da velocidade — não usada pelos wrappers atuais |
| `vdisp{40..260}` | 20 | Computado | Dispersão da componente vertical — idem |
| `cis{1..19}` | 19 | Computado | Cisalhamento entre alturas — idem |
| `dir{h}` (graus) | 11 | Coletado | Substituído por `dir{h}_sin/cos` |
| `press`, `humid`, `temp` | 3 | **Coletado** (estações micrometeorológicas) | Metadado ambiental — não alimenta as redes |

**Print das primeiras linhas** — CSV bruto (16/09/2021 18:00; atenção ao
jitter do instrumento que a regularização corrige):

| id | ws40 | ws100 | ws140 | v100 | dir100 | disp100 | vdisp100 | cis1 | cis19 | press | humid | temp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021-09-16 18:00:00 | 7,44 | 8,51 | 8,83 | 0,01 | 53,95 | 1,16 | 0,47 | 0,0200 | 0,0075 | 1012 | 78 | 27,6 |
| 2021-09-16 18:10:00.000004 | 7,45 | 8,64 | 8,93 | 0,01 | 52,19 | 1,04 | 0,50 | 0,0340 | 0,0015 | 1012 | 78 | 27,5 |
| 2021-09-16 18:19:59.999999 | 7,59 | 8,65 | 9,00 | −0,07 | 52,02 | 1,04 | 0,35 | 0,0270 | 0,0070 | 1012 | 78 | 27,5 |
| 2021-09-16 18:30:00.000003 | 7,64 | 8,86 | 9,32 | −0,11 | 53,65 | 0,96 | 0,45 | 0,0260 | 0,0060 | 1012 | 79 | 27,5 |
| 2021-09-16 18:39:59.999998 | 9,06 | 10,00 | 10,30 | 0,12 | 53,49 | 0,72 | 0,41 | 0,0240 | 0,0020 | 1012 | 79 | 27,4 |

**Visão de entrada dos modelos** (pós-`load_dataset`: grade regularizada,
dir → sin/cos, + hora cíclica, descartes da Tabela B):

| timestamp | ws40 | ws100 | ws140 | v40 | v100 | dir40_sin | dir40_cos | dir100_sin | dir100_cos | hour_sin | hour_cos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2021-09-16 18:00 | 7,44 | 8,51 | 8,83 | −0,13 | 0,01 | 0,809 | 0,588 | 0,809 | 0,588 | −1,000 | 0,000 |
| 2021-09-16 18:10 | 7,45 | 8,64 | 8,93 | −0,10 | 0,01 | 0,780 | 0,626 | 0,790 | 0,613 | −0,999 | 0,044 |
| 2021-09-16 18:20 | 7,59 | 8,65 | 9,00 | −0,16 | −0,07 | 0,784 | 0,621 | 0,788 | 0,615 | −0,996 | 0,087 |
| 2021-09-16 18:30 | 7,64 | 8,86 | 9,32 | −0,14 | −0,11 | 0,803 | 0,596 | 0,805 | 0,593 | −0,991 | 0,131 |
| 2021-09-16 18:40 | 9,06 | 10,00 | 10,30 | −0,07 | 0,12 | 0,808 | 0,589 | 0,804 | 0,595 | −0,985 | 0,174 |

> **⚠️ Implicação para o texto do artigo (novo cartão c-34):** a frase
> de "Preparação dos dados" no `03` (editada no Overleaf) diz que foram
> incluídos atributos de **dispersão, cisalhamento e cíclicos** — mas os
> modelos atuais **não usam disp/vdisp/cis** (Tabela B): só ws/v, dir
> em sin/cos e hora cíclica. Ou o texto é ajustado (descrever a entrada
> real, citando disp/cis como computados e não usados), ou os atributos
> voltam ao pipeline. Decisão do autor; a tabela acima é a fonte
> canônica para a tabela do dataset pedida pela Vanessa (`V-6ace8ce9`).

<a id="sec-1-4"></a>
### 1.4 Contagem de amostras 7632 × 7561 — `V-6ace7f52` — [ANÁLISE FECHADA 21/10]

> **Fechamento (21/10, verificação direta em `data/dataset.csv`):** a
> hipótese de duplicata **está descartada** — **zero timestamps
> duplicados**. Números definitivos:
>
> - **7561 linhas** (16/09/2021 18:00 → 08/11/2021 14:29), **0 células
>   NaN**, **0 duplicatas**;
> - grade de 10 min do **próprio intervalo** = **7611** → **50 amostras
>   ausentes (0,7%)** — lacunas são **linhas inteiras ausentes**
>   (indisponibilidade do instrumento), não valores vazios;
> - os **7632 da Vanessa = 53 dias cheios** — o intervalo real não é
>   alinhado a dia (começa 18:00, termina 14:29), então a conta por
>   dias cheios não se aplica;
> - parte dos timestamps é **fora da grade exata** (ex. minutos :29) —
>   bruto do instrumento, sem arredondamento no export;
> - **`dataset.csv` é o bruto da campanha**: não tem coluna `imputed` e
>   a imputação wavelet é ferramenta à parte
>   (`src/impute_wind_data.py` → `wind_data_imputed.csv`), que **não
>   alimenta o modelo** — fecha a [LACUNA] do doc de revisão §3.4.
>
> **Falta só a redação** (≈30 min) — texto pronto para o `03`:
> > "A série bruta da campanha contém 7561 amostras válidas de
> > 16/09/2021 (18:00) a 08/11/2021 (14:29). Do total de 7611 intervalos
> > de 10 min do período, 50 (0,7%) estão ausentes devido a
> > indisponibilidade pontual do instrumento; não há valores faltantes
> > dentro das amostras coletadas."

<a id="sec-1-5"></a>
### 1.5 Figuras legíveis — `V-6ace71fd` (P1) + `V-6ace992b` (fonte)

> **Terreno preparado (21/10):**
> - **Heatmap:** fontsizes ampliados no gerador
>   `reports/src/relatorio_resultados.py` (`plot_horizon_heatmap` /
>   `plot_horizon_r2_heatmap`: rótulos 19–21, anotações 20, colorbar
>   18–20) — no artigo a figura entra a `\linewidth` de coluna
>   (≈0,41× a largura original), então o tamanho **efetivo** fica
>   ≈8–9 pt (legível). A regeneração acontece junto da rodada final;
>   depois, `python scripts/articles_sync.py sync sbr`.
> - **Figura 1 (P1):** adotada a opção A — `width=0.5\linewidth` →
>   `\linewidth` no `03_Material_e_metodos.tex` (dobro do tamanho, P1
>   legível; recorte com P1 centralizado fica como plano B se ainda
>   estiver pequeno no PDF).

<a id="sec-1-6"></a>
### 1.6 [RESOLVIDO] Fix da cadeia automática (`sbsr_tables.py`) — 21/10

- O vigia `watch_retreino_itildeq.sh` quebrou no passo
  `scripts/sbsr_tables.py` (`ValueError: ... is not in the subpath of
  ...` — cabeçalho gerado com `round_dir.relative_to(PROJECT_ROOT)`
  sobre caminho relativo; log em
  `pipeline/tmp/watch_retreino_itildeq.log`).
- **Já está corrigido** (`.resolve()` no cabeçalho) e validado:
  `python scripts/sbsr_tables.py --round-dir
  data/results/report_8020_itildeq --skip-recompute --no-baselines`
  roda com rc=0 e produz as tabelas da canônica restaurada (TCN 0,519 /
  TCN-Bi 0,490 / LSTM 0,318 / LSTM-Bi 0,329). Nada a fazer — a cadeia
  está pronta para a rodada final.

---

<a id="sec-2"></a>
## 2. Atuados — falta intervenção humana (escrever / dissertar / escolher)

Respostas já verificadas com dados/código; o que falta é decisão ou
redação sua. Textos prontos no doc de revisão, seções indicadas.

| # | Item (thread) | O que já está feito | O que falta de você |
|---|---|---|---|
| 1 | ~~Escolha da rodada canônica~~ **RESOLVIDA (20/10)** | Decidida a **mista `8020-itildeq`** (TCN 0,519/TCN-Bi 0,490 com MSE; LSTM 0,318/Bi 0,329 com ITILDE-Q — números da execução 16/10; serão atualizados pela rodada final). Cadeia ajustada (§1.1) | — (rodada final aguarda o HPO do LSTM-Bi, §1.1) |
| 2 | "Tá correto?" — loss ( `V-6ace8d78`) | Pesos 0,7→1,3 corretos (`src/models/losses.py:15-25`); com a canônica escolhida, o texto deve dizer "LSTM ITILDE-Q / TCN MSE" — roteiro de conteúdo no bloco [METODO-LOSS] do `03` | Redigir com os números da rodada final (**P2**) |
| 3 | Entendimento do negócio (`V-6ace64b6`) | Overleaf (`a6ea1cf`) já trouxe contexto down-ramps/Vento Sul; esqueleto de 4 pontos no doc §3.1 caso falte algo | Validar se cobre o pedido; dissertar o que faltar (**P2**) |
| 4 | Ferramentas/versões (`V-6ace820a`) | Versões verificadas (Python 3.10, TF 2.21, PyWavelets 1.8 com algoritmos prontos, sklearn 1.7, statsmodels 0.15, Optuna 4.9); parágrafo pronto (doc §3.5) | Revisar e colar (**P1**) |
| 5 | Verbos pessoais (`F-6ace2c7f`, resolvido no Overleaf mas texto mantém) | Reescrita impessoal pronta (doc §1.1) | Aplicar (ou decidir manter) |
| 6 | "Arquitetura" (`F-6ace2e9e`) | Regra de troca definida (arquitetura = combinação estrutura+rede; rede neural = mecanismo) | Varrer o texto e aplicar o critério |
| 7 | Legenda "Fonte:" + EOSOLAR (`V-6ace641f`) | Bib local `arcilanVentoSul` **já é** o artigo do projeto EOSOLAR (Assireu et al. 2022, Energies 15(7):2319) | Conferir o bib no **Overleaf**; escolher formato da legenda (doc §3.3) |
| 8 | Conclusões: recapitular objetivo / "área de estudo" / validação cruzada (`F-6ace2e79`, `F-6ace2f14`, `F-6ace2fa9`) | 3 reescritas prontas (doc §5.1–5.3); diagnóstico correto: dado é só do P1; CV temporal existe (Roberts 2016, já no bib) | Revisar e aplicar (**P1**) |
| 9 | Citações de baselines (Apêndice D) | Referências identificadas: Box & Jenkins (ARIMA), Breiman 2001 (RF), Hyndman & Athanasopoulos (persistência) | Criar entradas com dados verificados (skill `bibtex-articles`) (**P1**). **GRU não será abordada** (decisão 21/10) — resta só `\cite{vaswani2017attention}` na menção a Transformers nas Conclusões (ex-§5.5 do doc; chave já no bib) |

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
- **Onde já está:** `reports/src/relatorio_resultados.py`
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

## 5. Ordem de execução

Substituída pela **§6** (plano consolidado com esforço × importância ×
prioridade, considerando o HPO do LSTM-Bi em andamento).

---

<a id="sec-6"></a>
## 6. Plano para zerar as pendências (21/10) — esforço × importância × prioridade

Contexto que molda o plano: o **LSTM-Bi está em Optuna** (resgatar a
ITILDE-Q) ⇒ **Resultados/Discussão, tabela de hiperparâmetros, parágrafo
da loss e números do Resumo podem mudar** ⇒ tudo que depende de números
vira **P2** (executar só depois da rodada final). O que independe de
números (**P1**) é executável agora; **P3** é fechamento de
pré-submissão.

Prioridade = quadrante (esforço ↓, importância ↑). **Depende = treino** ⇒
aguardar o HPO do LSTM-Bi fechar.

| # | Item | Esforço | Importância | Prioridade | Depende |
|---|---|---|---|---|---|
| 0 | ~~Fix `sbsr_tables.py` (cabeçalho com path relativo)~~ **✓ já corrigido e validado (§1.6)** | 10 min | Alta | — | — |
| 1 | §1.4 Contagem: redigir a correção no `03` (análise fechada, texto pronto) | 30 min | Alta | **P1 — agora** | — |
| 2 | §2.9 Bib baselines (Box & Jenkins, Breiman, Hyndman) + `\cite{vaswani2017attention}` (GRU fora — decisão 21/10) | 45 min | Média | **P1 — agora** | — |
| 3 | §2.4 Bibliotecas/versões (parágrafo pronto, revisar e colar) | 15 min | Média | **P1 — agora** | — |
| 4 | §2.8 Conclusões: aplicar as 3 reescritas prontas | 30 min | Alta | **P1 — agora** | — |
| 5 | §1.5 Figuras: **gerador já ajustado** (fontsizes heatmaps, fig. 1 `\linewidth`) — falta regenerar + resync | 1 h | Alta | P1 (código ✓) → regen **P2** | regen→treino |
| 6 | §2.3 Negócio: validar o que o Overleaf já cobre | 15 min | Média | P2 | — |
| 7 | §2.2 Loss: redigir estrutura ("LSTM ITILDE-Q / TCN MSE", sem números) | 15 min | Alta | P2 (finalizar pós-treino) | números→treino |
| 8 | §1.1 Rodada final: treino mista com HPs novos do HPO → regen **com baselines** → var. implícita única (critério de pronto) | 1 h humano + ~30 min máquina | **Crítica** (fecha o [CRÍTICO]) | **P2 — quando o HPO fechar** | HPO LSTM-Bi |
| 9 | §1.2 DM no artigo: parágrafo (H₀ + HAC/blocos) + tabela de p-valores (`mcs.json`) + já ok: 0,08/0,04 corrigidos no Overleaf | 1 h | **Crítica** (pedido da Vanessa) | P2 | item 8 |
| 10 | §1.3 Tabelas de hiperparâmetros + dataset (gerar de JSONs; TCN já, LSTM após HPO) | 1 h | Alta | P2 | HPO (linhas LSTM) |
| 11 | Integração completa (`sbsr-integrate`): tabelas `tab:error_*` + Resumo + loss + consistência (figura = tabela da mesma rodada) | 1–2 h | **Crítica** | P2 | itens 8–10 |
| 12 | Push Overleaf + sync de figuras | 10 min | Alta | P2 | item 11 |
| 13 | Remover blocos `% COMENTÁRIOS DE REVISÃO` + `consideracoes_extras.tex` | 30 min | Alta (bloqueia submissão) | **P3 — fechamento** | item 11 |
| 14 | Validação final do PDF no Overleaf (floats, bib, legibilidade das figuras, var. implícita única) | 30 min | Alta | **P3 — fechamento** | item 13 |

**Totais:** ≈8–10 h humanas até a submissão, sendo **≈2,5 h de P1
executável agora** (itens 1–4, sem risco de ficar obsoleto com o HPO) e
**≈4 h de críticos** (itens 8, 9 e 11) no caminho crítico real.

### Fases

1. **Fase 0 — hoje, em paralelo ao Optuna (≈2,5 h):** itens 1–6 + estrutura
   do item 7 (sem números). Nada aqui é invalidado pelo HPO.
2. **Fase 1 — quando o HPO do LSTM-Bi fechar (≈1,5 h ativa):** item 8
   (treino mista com os HPs novos; backup antes; se LSTM-Bi melhorar,
   o escopo TCN vs LSTM do artigo se fortalece) → regeneração com
   baselines na mesma grade → checar `MSE/(1−R²)` igual nas 7 linhas →
   `mcs.json` pronto → regen das figuras (fontsizes novos já no gerador).
3. **Fase 2 — integração (≈3–4 h):** itens 9–12 (DM, tabelas HPs/dataset,
   `sbsr-integrate`, push Overleaf).
4. **Fase 3 — fechamento (≈1 h):** itens 13–14 → **pronto para submeter**.

### Comandos da Fase 1 (cadeia já ajustada)

```bash
cp -r pipeline/tmp/trained_8020_itildeq \
      pipeline/tmp/trained_8020_itildeq_backup_$(date +%Y%m%d%H%M)
python scripts/experiments/lidar_8020_itildeq_train.py          # HPs/losses por modelo
python reports/src/relatorio_resultados.py \
  --models-dir pipeline/tmp/trained_8020_itildeq \
  --out-dir data/results/report_8020_itildeq        # SEM --no-baselines
python scripts/sbsr_tables.py \
  --round-dir data/results/report_8020_itildeq
python scripts/articles_sync.py sync sbr
python reports/src/builder.py build
```
