# Avaliação Comparativa de Ferramentas AutoML em Problemas de Regressão

Código, resultados brutos e análise estatística de um Trabalho de Conclusão de Curso (Ciência da Computação, IFSULDEMINAS – Câmpus Muzambinho) que compara quatro ferramentas AutoML de código aberto (**AutoGluon, FLAML, H2O AutoML e TPOT**) em **12 conjuntos de dados públicos de regressão**.

## Protocolo experimental
- Regressão com alvo único; métrica-guia **RMSE** (também MAE e R²).
- Partição 80/20 com `random_state = semente`; **5 repetições** (sementes 42 a 46); a semente também é passada à ferramenta.
- Orçamento de busca de **300 s** por execução, igual para todas as ferramentas e conjuntos.
- Ambiente: Google Colab gratuito (2 vCPUs, 12,67 GB de RAM, GPU T4 disponível), Python 3.13.
- Estatística: teste de Friedman sobre o RMSE médio por conjunto de dados, com pós-teste de Nemenyi (Demšar, 2006). MAE, R², mediana e outras variações são análises de sensibilidade exploratórias.
- 4 ferramentas × 12 conjuntos × 5 sementes = 240 execuções previstas; **217 válidas**. As execuções sem resultado estão registradas em `resultados/*.falha`; uma ferramenta sem nenhuma execução válida em um conjunto recebe a pior posição naquele conjunto.

Conjuntos de dados: Concrete, Abalone, Mercedes-Benz, Santander, Wine Quality, Kin8nm, Brazilian Houses, Bike Sharing, California Housing, News Popularity, Diamonds e Black Friday (obtidos do OpenML e do scikit-learn; ver `common/datasets_meta.py`).

## Estrutura
```
common/            código-fonte
  runner.py          executa as ferramentas e grava um JSON por execução
  datasets_meta.py   carga dos conjuntos de dados
  diagnostico.py     registro do ambiente (versões, GPU)
  consolidar.py      resumo rápido dos resultados
  analise_final.py   tabelas, testes estatísticos e figuras
  analise_brazilian.py, analise_autogluon_diagnostico.py   análise complementar do Brazilian Houses
notebooks/         executar_experimentos.ipynb (coleta no Colab, retomável) e diagnostico_autogluon_brazilian.ipynb
resultados/        um JSON por execução ({ferramenta}__{dataset}__seed{n}.json), previsões (.npz) e falhas (.falha)
analise_final/     tabelas (CSV e XLSX), figuras (PNG) e LEIA-ME_resultados.md com a leitura dos resultados
```

## Como reproduzir a análise
Com os resultados já incluídos em `resultados/`:
```bash
pip install -r requirements.txt
python common/analise_final.py
python common/analise_brazilian.py
```
As tabelas e figuras são geradas novamente em `analise_final/`.

## Como reproduzir os experimentos
Os experimentos foram executados no Google Colab com o notebook `notebooks/executar_experimentos.ipynb`: ele instala as versões fixas das ferramentas (ver `requirements.txt`), obtém o código deste repositório, monta o Google Drive e chama `common/runner.py`. Os conjuntos de dados, as ferramentas e as sementes são definidos na primeira célula. O runner pula as execuções que já têm arquivo `.json` ou `.falha`, de modo que a coleta é retomável, e o próprio notebook traz as instruções para dar continuidade ao estudo (novos conjuntos de dados ou novas ferramentas).

## Observações importantes
- **Medição de memória:** na primeira etapa da coleta (8 conjuntos), as ferramentas rodaram no mesmo processo, e o aumento de RAM registrado cresce com a ordem de execução. Por isso, as análises de memória usam somente as 29 execuções feitas em processo isolado (runner com um processo por execução); a lista das execuções não isoladas está em `resultados/rodada2_execucoes_nao_isoladas.txt`.
- **Limite de segurança de 1.200 s:** no runner mais recente, cada execução roda em um processo separado, que é encerrado se passar de 1.200 s no total. Esse valor não é o orçamento de busca (300 s) e serve apenas para encerrar execuções travadas.
- **GPU:** a GPU esteve disponível em todas as execuções, mas somente o H2O a utilizou (aumento de memória da GPU acima de 150 MB).
- **Versões:** as versões das ferramentas foram as mesmas em toda a coleta. A única biblioteca que variou foi o PyTorch (2.11.0 e 2.14.1); os registros de ambiente estão em `resultados/ambiente*.json`.
- **Ferramentas não avaliadas:** o Auto-sklearn não pôde ser instalado no Python 3.13 (a versão 0.15.0 exige scikit-learn entre 0.24 e 0.25).

## Licença
Nenhuma licença foi definida. Sem uma licença explícita, todos os direitos são reservados ao autor; adicione um arquivo `LICENSE` se desejar permitir o reuso.
