# Classificação de Doenças de Pele - UNIFOR

Sistema de processamento e classificação de imagens dermatológicas para análise de lesões de pele usando deep learning.

## 📋 Sobre o Projeto

Este projeto implementa um pipeline completo de pré-processamento de imagens médicas para classificação de lesões de pele, com foco no dataset HAM10000. Inclui ferramentas para:

- Aplicação de máscaras de segmentação
- Correção de vinheta (iluminação não-uniforme)
- Normalização de cor e contraste
- Data augmentation
- Preparação de imagens para modelos de deep learning

## 📁 Estrutura do Projeto

```
ClassificacaoDoencasDePeleUnifor/
├── scripts/                          # Scripts organizados por categoria
│   ├── preprocessing/                # Pré-processamento geral
│   │   ├── circle_to_square.py      # Extração de ROI circular
│   │   └── color_constancy_clahe.py # Normalização de cor
│   ├── augmentation/                 # Data augmentation
│   │   └── augment_skin_images.py   # Geração de augmentations
│   └── ham10000_pipeline/            # Pipeline HAM10000
│       ├── apply_masks.py           # Aplicação de máscaras + correção vinheta
│       └── resize_for_model.py      # Redimensionamento para modelos
├── HAM10000/                         # Dataset HAM10000
│   ├── images/                       # Imagens originais
│   ├── masks/                        # Máscaras de segmentação
│   └── model_ready_*/                # Imagens prontas para treino
└── requirements.txt                  # Dependências Python
```

## 🚀 Pipeline Recomendado (HAM10000)

```bash
# 1. Aplicar máscaras + correção de vinheta
python scripts/ham10000_pipeline/apply_masks.py
# → Escolha modo 4 (Recortado + Vinheta corrigida)

# 2. Redimensionar para modelo
python scripts/ham10000_pipeline/resize_for_model.py
# → Escolha 224×224 com padding

# 3. Treinar modelo
# Usar imagens de: HAM10000/model_ready_224x224_padded/
```

## Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DA_PASTA_DO_PROJETO>
    ```

2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    # Criar o ambiente
    python -m venv venv

    # Ativar no Windows
    .\venv\Scripts\activate

    # Ativar no macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    O arquivo `requirements.txt` contém todas as bibliotecas necessárias. Instale-as com o seguinte comando:
    ```bash
    pip install -r requirements.txt
    ```

## 📚 Documentação dos Scripts

Cada categoria de scripts possui documentação detalhada:

- **[scripts/README.md](scripts/README.md)** - Visão geral e fluxo de trabalho
- **[scripts/preprocessing/](scripts/preprocessing/)** - Scripts de pré-processamento geral
- **[scripts/augmentation/](scripts/augmentation/)** - Scripts de data augmentation
- **[scripts/ham10000_pipeline/](scripts/ham10000_pipeline/)** - Pipeline completo HAM10000

## 🔧 Scripts Disponíveis

### Pipeline HAM10000 (Principal)
```bash
# Aplicar máscaras + correção de vinheta
python scripts/ham10000_pipeline/apply_masks.py

# Redimensionar para modelo
python scripts/ham10000_pipeline/resize_for_model.py
```

### Pré-processamento Geral
```bash
# Extrair ROI circular
python scripts/preprocessing/circle_to_square.py

# Normalização de cor (Gray-World + CLAHE)
python scripts/preprocessing/color_constancy_clahe.py
```

### Data Augmentation
```bash
# Gerar augmentations
python scripts/augmentation/augment_skin_images.py
```

Todos os scripts são interativos e guiarão você através das opções disponíveis.

## 🧠 Classificador em Ensemble

O pipeline de inferência usa um **ensemble hierárquico**: um modelo generalista (`ResNet-RS50` com 3 saídas) gera probabilidades globais e três especialistas binários refinam as classes em pares. Todo o fluxo está em `scripts/classification/service.py`.

Pesos do ensemble (otimizados via algoritmo genético):

- Generalista: `0.6280`
- Benignos × Malignos: `0.2073`
- Malignos × Pré-Malignos: `0.1646`
- Pré-Malignos × Benignos: `9.76e-05`

### Como cada modelo foi construído

**Generalista (ResNet-RS50 3 classes)**

- Backbone: ResNet-RS50 do TIMM pré-treinado em ImageNet 1K.
- Ajustes: substituição da camada fully connected por `Linear(2048 → 3)`; dropout 0,3 antes da cabeça.
- Treino: imagens HAM10000 balanceadas com oversampling, augmentations geométricas leves (flip horizontal, rotação ±15°, zoom) e ajuste de cor via ColorJitter.
- Otimização: AdamW (lr 1e-4, weight decay 1e-4) com ReduceLROnPlateau e early stopping em 15 épocas.
- Normalização: mean/std de ImageNet e redimensionamento 448×448 com padding para preservar aspecto.
- Saída: vetor `[P(benigno), P(maligno), P(pré-maligno)]` sem softmax aplicado fora do modelo (função de serviço aplica `torch.softmax`).

**Especialistas binários (ResNet-RS50 para pares)**

- Pares cobertos: `Benignos×Malignos`, `Malignos×Pré-Malignos`, `Pré-Malignos×Benignos`.
- Arquitetura: mesma base ResNet-RS50, mas última camada substituída por `Linear(2048 → 2)`; dropout 0,4 para melhorar generalização.
- Dados: subconjuntos derivados do HAM10000 balanceados via `scripts/augmentation/augment_skin_images.py` (ESM), com validação estrita estratificada 80/20.
- Augmentations: cortes randômicos 10%, mixup opcional (α=0,2) e ajustes suaves de cor; intuito é capturar texturas específicas dos pares.
- Otimizador: SGD com momentum 0,9, lr inicial 3e-4, ciclo cosine annealing por 30 épocas.
- Saída: probabilidade do primeiro rótulo do par (ex.: `p_maligno` ao comparar Malignos×Pré-Malignos); a função de serviço mapeia o índice para o rótulo correspondente.

**Roteamento dentro do serviço**

- Ensemble: combina as probabilidades dos especialistas diretamente com o vetor generalista conforme pesos anteriores.
- Cascade: calcula confiança máxima do generalista; se < limiar (`service.CASCADE_THRESHOLD`), consulta o especialista do par relativo à classe top-2 e mistura os pesos definidos por `set_cascade_weights`.
- Direct-specialist: identifica o par mais provável (top-1 × segundo lugar) e repondera apenas esse especialista com pesos definidos via `set_direct_weights`.
- Todas as saídas são renormalizadas para garantir soma 1. Metadados de qual especialista foi acionado acompanham o dicionário retornado por `classify_image_bytes`.

Passos em produção:

1. A imagem (normalizada com mean/std do ImageNet) passa pelo generalista `Treinamento Modelos/resnetrs50/best_model.pth`, gerando `p_general`.
2. Cada especialista localizado em `Treinamento Modelos/especialistas/resnetrs50/<par>` gera `p_especialista` para seu par de classes.
3. As probabilidades são combinadas pelo peso definido para o especialista correspondente, renormalizadas e retornadas com a indicação de quais especialistas contribuíram na decisão.

Para reproduzir a avaliação, execute:

```bash
python tests/test_classify.py
```

Esse comando gera matriz de confusão normalizada, relatório de classificação e salva os artefatos em `results/`.

## 🧪 Plano de Testes e Automação

Com a introdução dos especialistas binários, estabelecemos três modos oficiais de inferência e um fluxo completo para avaliá-los e calibrar seus pesos automaticamente. Tudo está descrito abaixo com o máximo de detalhe para facilitar reproduções e auditorias.

### Modos de inferência

1. **ensemble** – combina o modelo generalista com os três especialistas, ponderando as probabilidades finais para todas as classes.
2. **cascade** – usa o generalista como primeiro estágio; somente quando a confiança cai abaixo do limiar a predição é reavaliada por um especialista focado no par de maior probabilidade.
3. **direct-specialist** – identifica o par de classes mais provável via generalista e delega a decisão final ao especialista correspondente, ponderando generalista × especialista diretamente.

Cada modo ajusta pesos distintos (4 parâmetros no ensemble, 2 parâmetros nos demais) e grava seus artefatos em pastas exclusivas para evitar conflito de resultados.

### Conjunto de validação

- Raiz esperada: `~/data/val` (pode ser alterada via `--dataset-root`).
- Estrutura mínima:
    - `Benignos/*.jpg`
    - `Malignos/*.jpg`
    - `Pre-Malignos/*.jpg`
- Qualquer extensão listada em `tests/test_classify.py` (`.png`, `.jpg`, `.jpeg`, `.bmp`) é suportada.

### Execução dos três testes principais

Use o utilitário `tests/run_modes.py` para avaliar todos os modos em sequência, reaproveitando os mesmos bytes carregados do disco:

```bash
python tests/run_modes.py \
    --modes ensemble cascade direct-specialist \
    --dataset-root /caminho/para/val \
    --results-root results
```

- Saídas:
    - `results/ensemble_test/`
    - `results/cascade_test/`
    - `results/direct-specialist_test/`

Em cada pasta são gravados: `classification_report.txt`, `validation_confusion_report.png`, matriz de confusão em memória e o histórico impresso no terminal. Ajuste `--modes` para rodar subconjuntos ou `--results-root` para redirecionar a escrita.

### Tuning automático via Algoritmo Genético

`tests/tune_ensemble_weights.py` implementa o GA genérico. Ele recebe:

- `--mode {ensemble|cascade|direct-specialist}` – define o número e o significado dos pesos.
- `--population`, `--generations`, `--mutation-rate`, `--mutation-scale`, `--seed` – hiperparâmetros do GA.
- `--output-dir` – pasta onde salvar relatórios e o arquivo `<modo>_tuning_summary.json`.

Exemplo para o ensemble apenas:

```bash
python tests/tune_ensemble_weights.py \
    --mode ensemble \
    --population 24 \
    --generations 10 \
    --mutation-rate 0.6 \
    --mutation-scale 0.08 \
    --seed 123 \
    --output-dir results/ga_runs/ensemble_ga
```

`tests/tune_all_modes.py` orquestra o mesmo processo para os três modos em sequência, reutilizando os mesmos hiperparâmetros para garantir comparabilidade de resultados:

```bash
python tests/tune_all_modes.py \
    --modes ensemble cascade direct-specialist \
    --population 24 \
    --generations 10 \
    --mutation-rate 0.6 \
    --mutation-scale 0.08 \
    --results-root results/ga_runs
```

Cada execução gera:

- `results/ga_runs/<modo>_ga/`
    - `classification_report.txt` (melhor configuração encontrada)
    - `validation_confusion_report.png`
    - `<modo>_tuning_summary.json` (contém histórico do GA, pesos vencedores, parâmetros usados e caminhos dos artefatos)

### Fluxo sugerido

1. **Avaliar pesos atuais** – `python tests/run_modes.py` (certifica baseline).
2. **Rodar GA** – `python tests/tune_all_modes.py` ou chamadas individuais com `tune_ensemble_weights.py` para cada modo.
3. **Revalidar** – repetir `run_modes.py` restrito aos modos em que os pesos foram atualizados, confirmando que a acurácia reportada pelo GA se mantém.

Esse fluxo garante rastreabilidade completa entre pesos, métricas e artefatos de cada estratégia de inferência.