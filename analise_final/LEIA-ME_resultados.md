# Análise final — resultados e leitura (12 datasets)

Estudo: 4 ferramentas AutoML × **12 datasets** × 5 seeds (42–46), orçamento de 300 s, Colab gratuito. **217 de 240 execuções válidas** (18 falharam e 5 não foram tentadas). Etapa inicial: 8 datasets (158/160); etapa de ampliação: +4 datasets (Kin8nm, News Popularity, Santander, Black Friday; 59/80). Tudo é regerado por `common/analise_final.py` e `common/analise_brazilian.py` a partir de `resultados/`. A análise anterior (8 datasets) está preservada em `analise_final_8datasets_backup/`.

## O que há aqui
| Arquivo | Conteúdo |
|---|---|
| `tabelas_monografia.xlsx` | abas prontas para colar no Word: RMSE, MAE, R², Ranking, Testes, Nemenyi, Sensibilidade, Custo, GPU, Síntese, Datasets, Falhas, Execuções válidas, RAM (grupos isolados) |
| `tabelas/*.csv` | mesmas tabelas em CSV (versões `_numerico` têm média, dp e n separados) |
| `figuras/*.png` e `*.pdf` | 300 dpi, rótulos em português, mesma cor por ferramenta em todas |

## Regras adotadas (definidas antes do teste)
- **Célula sem nenhuma execução válida = pior posição do dataset** (empate entre as ferramentas que falharam). Células parciais usam a média das seeds válidas (n indicado nas tabelas).
- **Sensibilidade extra:** só os 10 datasets em que as 4 ferramentas produziram resultado (sem Santander e News Popularity) e só os 8 datasets da etapa inicial.
- **RAM:** só vale a medida das **29 execuções em processo isolado** (etapa de ampliação, runner v2). Rodada 1 inteira e 30 execuções da 1ª sessão da etapa de ampliação rodaram no mesmo processo e têm o aumento de RAM contaminado pela ordem (lista em `resultados/rodada2_execucoes_nao_isoladas.txt`).

## Resultados principais (verificados)
**Análise principal: Friedman sobre o RMSE médio por dataset (N = 12).**
- Ranking médio (1 = melhor): **H2O 2,00 · FLAML 2,33 · AutoGluon 2,38 · TPOT 3,29**.
- χ² = 6,68 (gl = 3), **p = 0,083** (Iman-Davenport: F = 2,51, p = 0,076). **H0 não é rejeitada.** Nemenyi informativo: DC = 1,35, maior diferença (TPOT − H2O) = 1,29, nenhum par difere. Com 8 datasets era p = 0,098.
- Vezes em 1º (RMSE): H2O 5, AutoGluon 4, TPOT 2 (Mercedes-Benz, News Popularity n = 2), FLAML 1 (Black Friday). FLAML foi 2º em 7 dos 12.

**Sensibilidade (exploratória, sem correção para múltiplos testes):**
| Análise | Ranking (AG / FLAML / H2O / TPOT) | p (Friedman) | Nemenyi |
|---|---|---|---|
| RMSE médio (principal, N = 12) | 2,38 / 2,33 / **2,00** / 3,29 | 0,083 | nenhum par |
| RMSE mediano | **2,13** / 2,25 / 2,17 / 3,46 | 0,030 | nenhum par |
| MAE médio | 2,29 / **2,08** / **2,08** / 3,54 | 0,013 | FLAML × TPOT e H2O × TPOT (p = 0,029) |
| R² médio | 2,29 / 2,33 / **1,92** / 3,46 | 0,022 | H2O × TPOT (p = 0,018) |
| sem Brazilian Houses (N = 11) | 2,23 / 2,36 / **2,09** / 3,32 | 0,103 | — |
| só datasets sem falha (N = 10) | 2,10 / 2,40 / **2,00** / 3,50 | 0,036 | H2O × TPOT (p = 0,046) |
| só etapa inicial (N = 8) | 2,13 / 2,63 / **1,88** / 3,38 | 0,098 | — |

Leitura dos resultados: **H2O, FLAML e AutoGluon são tecnicamente empatados**; **TPOT é o último em todas as análises** e só se separa significativamente do H2O (e do FLAML no MAE) em análises exploratórias. RMSE relativo ao melhor (média geométrica): H2O 1,05 · FLAML 1,11 · AutoGluon 1,24 · TPOT 1,53; sem Brazilian Houses: AutoGluon 1,01 · FLAML 1,04 · H2O 1,05 · TPOT 1,39.

**Execuções e falhas (`tabelas/falhas.csv`, `execucoes_validas_por_celula.csv`):** válidas AutoGluon 49 · FLAML 60 · H2O 60 · TPOT 48.
- AutoGluon: News Popularity (seeds 42–45 falharam por "Not enough time left", após 565–1.342 s; seed 46 não tentada), Santander (42–44 encerradas no limite rígido de 1.200 s; 45–46 não tentadas), Black Friday seed 45 (erro interno "Learner is already fit", 58 s, causa não investigada).
- TPOT: 10 falhas (Diamonds/45, Mercedes/46 da etapa inicial; Santander 42–44, News 43–45, Black Friday 43 e 45), mensagem "nenhum indivíduo avaliado na população inicial" (9 delas por max_eval_mins); Santander 45–46 não tentadas.
- FLAML e H2O: 60/60. As 5 execuções "não tentadas" foram omitidas após 3 falhas para poupar tempo de sessão e tratadas como falha (suposição).

**Estabilidade:** CV mediano do RMSE entre seeds — AutoGluon 3,5% · H2O 3,5% · FLAML 4,7% · **TPOT 15,8%**.

**Custo computacional** (aba *Custo computacional*):
- Tempo real médio (execuções válidas): H2O 319 s · AutoGluon 329 s · FLAML 331 s · TPOT 350 s. Execuções > 330 s: H2O 10,0%, FLAML 18,3%, AutoGluon 22,4%, **TPOT 62,5%**. Maiores: FLAML 734 s (Bike Sharing/44) e 702 s (Black Friday/42). As falhas do AutoGluon (565–1.342 s e 1.200 s) não entram nessas médias.
- **Memória (29 execuções isoladas: AG 4, FLAML 9, H2O 10, TPOT 6):** mediana do aumento de RAM AutoGluon 682 MB · FLAML 836 MB · TPOT 1.367 MB · **H2O 1.611 MB** (TPOT ≈ 2,0× e H2O ≈ 2,4× o AutoGluon). Pico mediano 1,33 / 1,51 / 2,06 / 2,29 GB. Em 4 pares dataset–seed com as 4 ferramentas isoladas, H2O e TPOT superaram AutoGluon e FLAML em 3 (exceção: Black Friday/44). **Os valores de 2,2–2,5× da versão anterior da monografia eram contaminados e foram descartados.**
- **GPU: só o H2O usou** (60/60 execuções, 182–1.128 MB). As demais ficaram em 0 com limiar de 150 MB; 21 execuções (2 AutoGluon, 13 FLAML, 6 TPOT) marcaram exatamente 102 MB, sobrecarga de inicialização.
- Trade-off (`fig07`): H2O tem o melhor ranking com o maior consumo de memória; AutoGluon e FLAML consomem menos com ranking equivalente; TPOT é pior em desempenho e estabilidade.

**Brazilian Houses (achado do estudo):**
- **O alvo é (quase) a soma de quatro atributos.** `total (BRL)` = `hoa + rent_amount + property_tax + fire_insurance`: a diferença é ≤ R$ 1 em 90,4% das linhas e ≤ R$ 20 em 99,9% (correlação 1,000). O dataset traz o alvo "embutido" nas variáveis. Ferramentas que aprendem uma relação linear (soma) podem acertá-lo quase exatamente; árvores saturam nos valores extremos. O TPOT teve RMSE de 1 a 10 em 3 das 5 seeds, provavelmente por ter encontrado esse tipo de relação (não verificado: os pipelines do TPOT não foram salvos).
- **Atributos com valores extremos (prováveis erros de digitação):** área de até 46.335 m² (mediana 90), condomínio (`hoa`) de até R$ 1.117.000 e IPTU (`property_tax`) de até R$ 313.700. É daí que vêm o aluguel total de R$ 1.120.000 (duas linhas idênticas) e os demais alvos altos.
- **Duplicatas:** 358 linhas (3,3%); ~94–112 linhas do teste por seed são cópias do treino. O FLAML reproduz o valor de R$ 1,12 milhão exatamente (memorização). **As duplicatas explicam parte da instabilidade de FLAML e TPOT, mas não o desempenho do AutoGluon.**
- **AutoGluon (execução de diagnóstico, que reproduziu o estudo: RMSE 16.636 contra 17.182 na seed 44):**
  - O ensemble final colapsa na rede neural `NeuralNetFastAI`: `WeightedEnsemble_L2` e `L3` têm exatamente o mesmo `score_val` dela, e as métricas de teste são idênticas. Ela teve a **melhor nota de validação** (RMSE 9.854, contra 10.866 ou mais nos demais modelos, na seed 45), com todas as notas na faixa de 10 a 12 mil, ou seja, num RMSE de validação dominado por poucas linhas extremas.
  - No teste, a rede **extrapola** em linhas com atributo extremo: previu R$ 1.047.703 para um imóvel de R$ 316.900 (IPTU = R$ 313.700), R$ 219.810 para um de R$ 10.220 (área = 46.335 m²) e R$ 123.935 para um de R$ 18.320 (área = 24.606 m²). Esse único erro (R$ 730.803) responde por **90% do erro quadrático**; sem ele, o RMSE cairia de 16.636 para ≈ 5.200.
  - ExtraTrees e RandomForest quase não superestimam (≤ R$ 3.447), mas **subestimam** os extremos; LightGBM e CatBoost ficam no meio (superestimativas de R$ 39 mil a R$ 293 mil).
  - O empilhamento **não** é a causa: os níveis L2 e L3 do ensemble repetem a mesma rede.
  - **Variantes testadas (seeds 44 e 45; `fig11_autogluon_diagnostico`, `tabelas/diagnostico_autogluon_*.csv`):**
    - *Sem rede neural e KNN:* a maior superestimativa cai de R$ 635–731 mil para R$ 890–4.004; o RMSE cai 32–38% (16.636 → 10.348 e 13.834 → 9.400) e o MAE cai ≈ 70% (1.315 → 401 e 1.200 → 372). **Confirma que a rede é a origem das superestimativas.**
    - *Alvo em log1p:* a superestimativa some e o MAE cai 52–64%, mas o RMSE **não melhora de forma consistente** (−22% na seed 44, +21% na seed 45), porque passa a ser dominado pelas linhas de R$ 1,12 milhão, que o modelo subestima. Não é uma correção geral.
    - Mesmo sem a rede, o RMSE (9.400 a 10.348) continua acima do de FLAML (6.484 e 6.723) e H2O (2.932 e 1.133) nessas seeds, então a posição do AutoGluon no ranking do Brazilian Houses não muda. Já o MAE (372 a 401) fica na mesma faixa das outras ferramentas (FLAML 230 e 317; H2O 443 e 182): o RMSE residual vem de poucas linhas extremas que as árvores subestimam. Excluindo as linhas duplicadas do teste, o RMSE das variantes fica em ≈ 6.600–7.200.
    - Limitação: 2 seeds, uma execução por variante. As variantes **não entram** no ranking principal, que segue o protocolo (configuração padrão para todas as ferramentas).
- **Leitura:** neste dataset o RMSE mede, principalmente, a capacidade de representar uma soma e de extrapolar com segurança para atributos extremos; isso favorece quem tem modelos lineares ou saturantes e penaliza a rede neural escolhida pelo AutoGluon. A escolha da rede, feita por uma validação sem poder de discriminação (diferença de 8% a 20% num RMSE dominado por poucas linhas), é uma **fragilidade do processo de seleção do AutoML**, não um defeito geral da ferramenta (em 6 dos 10 datasets em que obteve resultado o AutoGluon fica entre as duas melhores posições). O dataset é tratado como caso especial, e a análise é apresentada com e sem ele.

## Diferenças em relação ao projeto de pesquisa original
1. **Repetições:** projeto previa seed 42 única; foram usadas 5 seeds (42–46), por recomendação da banca (repeated holdout / validação Monte Carlo; Aragão et al., 2025, usam 20 seeds).
2. **Orçamento:** 600 s no piloto → **300 s** na comparação final. O piloto não entra na análise.
3. **Ferramentas:** projeto cita AutoGluon e FLAML como base; foram adicionadas **H2O e TPOT**. Auto-sklearn **não pôde ser instalado**: v0.15.0 (última, set/2022) exige scikit-learn >= 0.24 e < 0.25, cuja compilação falhou no Python 3.13 (testado em Colab separado).
4. **Datasets:** 3 → 12, com critério (volume, atributos, categóricos, esparsidade); a ampliação para 12 foi recomendação da orientadora (poder do Friedman). Novos: Kin8nm (OpenML 189; o ID 807 é classificação), News Popularity (42724), Santander (42572), Black Friday (41540).
5. **Seção 4.3 do projeto afirma que a GPU T4 "foi utilizada"**: corrigir para "T4 disponível; uso medido apenas no H2O".
6. **Versões:** AutoGluon 1.5.0, FLAML 2.7.0 (o projeto cita 2.6.0), H2O 3.46.0.12, TPOT 1.1.0, CatBoost 1.2.10. Python 3.13.15 (seeds 42–43 da etapa inicial) e 3.13.16 (demais); PyTorch 2.11.0 (etapa inicial) e **2.14.1+cu130 (etapa de ampliação)**; mesmas versões das ferramentas em toda a coleta.
7. **TPOT:** configurado com `cv=3` e `max_eval_time_mins=1` (a configuração padrão estourava o orçamento: até 929 s para 300 s) e one-hot encoding das categóricas.
8. **Runner v2 (etapa de ampliação):** um processo por execução, limite rígido de 1.200 s, `.falha` por execução que não gerou resultado.

## Limitações para declarar
- 12 datasets → poder ainda limitado no Friedman. Um único hold-out por seed; as 5 partições se sobrepõem.
- Falhas (23 de 240) e regra "falha = pior posição"; 5 execuções omitidas tratadas como falha por suposição.
- Sensibilidades são exploratórias e sem correção de múltiplos testes; a análise oficial é o RMSE médio.
- RAM confiável só em 29 execuções isoladas; Colab gratuito (variação de tempo entre sessões); PyTorch diferente entre etapas.
- AutoGluon com preset `high_quality` e TPOT com parâmetros reduzidos; FLAML e H2O padrão.
- Previsões salvas (`_pred.npz`) existem para 198 das 217 execuções válidas.

## Possíveis continuações
- A análise está completa com as execuções disponíveis.
- Re-medir a RAM dos 8 datasets da etapa inicial em processo isolado (≈ 32 execuções, ≈ 3 h) para ter medida confiável em todos os conjuntos.
- Testar o AutoGluon com preset mais leve (`medium_quality`) e TPOT com mais tempo no Santander/News Popularity, para testar a hipótese de que as falhas decorrem do orçamento.
