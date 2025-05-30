## Estrutura do projeto

```
APS MPA/
├── ConvNN.py              # Definição e treinamento da rede neural convolucional (CNN)
├── MLP.py                 # Definição e treinamento da rede MLP fully-connected
├── RandomForest.py        # Implementação do classificador Random Forest usando Scikit-Learn
├── main.py                # Script principal para rodar os modelos e gerar métricas
├── metrics.py             # Geração de gráficos (ROC, matriz de confusão) e métricas detalhadas
├── data_loader.py         # Funções de carregamento e pré-processamento de MNIST e MNIST-C
├── README.md              # Documentação do projeto
├── requirements.txt       # Dependências necessárias para executar o projeto
│
├── data/                  # Dados originais do MNIST (baixados automaticamente pelo torchvision)
│
├── mnist_c/               # Versão corrompida do MNIST (MNIST-C), organizada por tipo de distorção
│   ├── brightness/
│   ├── shot_noise/
│   └── ...                # Outras corrupções (motion_blur, fog, etc.)
│
├── modelos/               # (Opcional) Diretório para salvar pesos dos modelos treinados
├── results/               # Gráficos, métricas e arquivos gerados durante os testes
└── .venv/                 # Ambiente virtual Python (opcional, mas recomendado)
```

## Resultados

As métricas utilizadas para avaliar os modelos foram:

- **Matriz de confusão**: tabela que relaciona as classes verdadeiras com as classes preditas, destacando acertos na diagonal principal e erros fora dela. Permite identificar padrões específicos de confusão entre classes.

- **Curvas ROC (One-Versus-All)**: conjunto de curvas ROC, uma para cada classe, em que cada classe é tratada como positiva e todas as outras como negativas. Mostra a capacidade do modelo de separar corretamente cada classe individualmente em termos de taxa de verdadeiros positivos e falsos positivos.

- **Acurácia**: número de acertos dividido pelo total de exemplos.

  Acurácia = (número de acertos) / (total de exemplos)

- **Macro F1-score**: é a média simples do F1-score calculado separadamente para cada classe. O F1-score combina precisão (quantos dos positivos previstos estão corretos) e revocação (quantos dos positivos reais foram identificados) em uma única métrica por classe. No macro F1, cada classe tem o mesmo peso, independentemente de quantas amostras ela tem.

  Macro F1 = (F1_1 + F1_2 + ... + F1_K) / K

- **Log Loss**: mede o quão bem calibradas estão as probabilidades atribuídas pelo modelo. Para cada exemplo, considera apenas a probabilidade que o modelo deu à classe correta. Previsões confiantes e corretas geram penalidades baixas; previsões erradas e confiantes geram penalidades altas.

  Log Loss = – (1/N) × Σ log(p_classe_correta)


### MNIST Padrão

O primeiro conjunto de experimentos foi realizado com a base MNIST original, composta por imagens de dígitos manuscritos (28×28 pixels, em tons de cinza), bem centralizadas e com pouco ruído. Os modelos treinados foram uma Rede Neural Convolucional (CNN), uma Rede Neural Feedforward (MLP) e uma Random Forest (RF). Todos os modelos foram avaliados com base em métricas clássicas de classificação multiclasse.

| Modelo           | Acurácia | Macro F1 | Log Loss |
|------------------|----------|----------|----------|
| **CNN**          | **0.991**    | **0.9909**   | **0.0268** ✅ |
| **MLP**          | 0.974    | 0.9733   | 0.0900 ⚠️ |
| **Random Forest**| 0.970    | 0.9702   | **0.2466** 🔴 |


A CNN apresentou desempenho superior em todas as métricas, com maior precisão e menor entropia (log loss). O MLP também obteve resultados robustos, embora com pequenas quedas em classes mais visuais como 2, 3 e 8. A Random Forest, apesar de competitiva em acurácia, mostrou-se menos calibrada — evidenciado por um log loss mais alto.

#### Análise Qualitativa

- CNN: Erros residuais concentrados em dígitos visualmente ambíguos (como 5 vs 3 ou 9 vs 4), mas com excelente generalização.

- MLP: Desempenho sólido, mas mais sensível a variações sutis de traço.

- RF: Forte capacidade de classificação geral, porém mais suscetível a confusões entre dígitos parecidos, especialmente em regiões limítrofes do espaço de decisão.

#### Gráficos gerados:

#### Gráficos gerados:

Matrizes de confusão e curvas ROC One-vs-All para cada modelo (disponíveis na pasta `/results`).

---

**CNN – MNIST**

<p align="center">
  <img src="https://github.com/user-attachments/assets/da8c8010-bfec-4efc-9d84-4e0135693e93" width="45%"/>
  <img src="https://github.com/user-attachments/assets/97fc724d-2147-427f-ac6a-5217366fa2fb" width="45%"/>
</p>

---

**MLP – MNIST**

<p align="center">
  <img src="https://github.com/user-attachments/assets/58d66996-3e5d-4fea-a271-ca56ff7f97ae" width="45%"/>
  <img src="https://github.com/user-attachments/assets/9af86c50-7a29-494f-9db1-c220bbca0e72" width="45%"/>
</p>

---

**Random Forest – MNIST**

<p align="center">
  <img src="https://github.com/user-attachments/assets/2fc2edbb-9545-4ca2-b590-c1a45fb8f1c3" width="45%"/>
  <img src="https://github.com/user-attachments/assets/e79f76d4-8f28-4d05-ab6a-a5e67f8b5357" width="45%"/>
</p>



### MNIST-C

Para avaliar a robustez dos modelos em condições adversas mais próximas de aplicações reais, testamos CNN, MLP e Random Forest sobre a base MNIST-C, que introduz distorções visuais sistemáticas aos dígitos manuscritos. Utilizamos as corrupções shot_noise e motion_blur, ambas com severidade 5, representando cenários difíceis, porém legíveis.

As métricas consideradas foram acurácia, macro F1-score e log loss (entropia cruzada), esta última sendo crucial para medir calibração de confiança dos modelos.

#### Corrupção: brightness (severidade 5)

| Modelo           | Acurácia | Macro F1 | Log Loss |
|------------------|----------|----------|----------|
| **CNN**          | **0.982**   | **0.9816**   | **0.0543** ✅ |
| **MLP**          | 0.927    | 0.9251   | 0.2637 ⚠️ |
| **Random Forest**| 0.971    | 0.9704   | **0.3017** 🔴 |

#### Corrupção: shot-noise (severidade 5)

| Modelo           | Acurácia | Macro F1 | Log Loss |
|------------------|----------|----------|----------|
| **CNN**          | **0.979**    | **0.9776**   | **0.0618** ✅ |
| **MLP**          | 0.955    | 0.9531   | 0.1606 ⚠️ |
| **Random Forest**| 0.955    | 0.9534   | **0.4516** 🔴 |


#### Corrupção: motion-blur (severidade 5)

| Modelo           | Acurácia | Macro F1 | Log Loss |
|------------------|----------|----------|----------|
| **CNN**          | **0.977**    | **0.9761**   | **0.0658** ✅ |
| **MLP**          | 0.956    | 0.9542   | 0.1388 ⚠️ |
| **Random Forest**| 0.964    | 0.9628   | **0.3317** 🔴 |

