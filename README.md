# CBIR — Recuperação de Imagens por Conteúdo

> **Content-Based Image Retrieval** com SIFT + Bag of Visual Words + similaridade espacial (IoU), aplicado ao dataset **Pascal VOC 2012** nas classes `cat` e `horse`.

**Autores:** Maurício Fabiano Azevedo · Eduardo de Almeida Barboza

---

## Índice

- [Visão Geral](#visão-geral)
- [Pipeline do Sistema](#pipeline-do-sistema)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação e Execução](#instalação-e-execução)
- [Dataset — VOC2012](#dataset--voc2012)
- [Hiperparâmetros](#hiperparâmetros)
- [Métricas de Avaliação](#métricas-de-avaliação)
- [Resultados](#resultados)
- [Limitações](#limitações)

---

## Visão Geral

Este projeto implementa um sistema de **recuperação de imagens por conteúdo (CBIR)** que, dada uma imagem de consulta, retorna as imagens mais similares de um índice. A similaridade é calculada combinando:

- **Similaridade visual** — descritores SIFT codificados como histogramas Bag of Visual Words (BoVW), comparados por cosseno.
- **Similaridade espacial** — Intersection over Union (IoU) entre as bounding boxes normalizadas dos objetos.

O score final de cada candidata é: `score = α × visual + (1 − α) × espacial`, onde `α = 0.8` por padrão.

---

## Pipeline do Sistema

```
Dataset VOC2012
      │
      ▼
parse_annotation()       ← lê XMLs, extrai labels e bboxes
      │
      ▼
load_dataset()           ← seleciona até 100 docs e 5 queries mais conectadas
      │
      ▼
object_proposals()       ← usa bboxes anotadas como regiões de interesse
      │
      ▼
sift_descriptors()       ← extrai descritores SIFT (N × 128) por região
      │
      ▼
build_codebook()         ← K-means com 300 palavras visuais
      │
      ▼
bovw()                   ← histograma normalizado de 300 dimensões por região
      │
      ▼
index_images()           ← armazena {bbox, BoVW, spatial} por imagem
      │
      ▼
run_query()              ← cosine similarity + IoU → top-K resultados
      │
      ▼
evaluate()               ← MAP, Precision@K, Recall@K
      │
      ▼
visualize_results()      ← salva ranking em PNG na pasta resultados/
```

---

## Estrutura do Projeto

```
CBIR_Recuperacao_de_imagem/
├── main.py                              # Código principal do sistema
├── documentacao_recuperacao_imagens.pdf # Documentação técnica detalhada
├── resultados/                          # Rankings gerados (PNGs)
├── .gitignore
└── README.md
```

> **Nota:** o dataset VOC2012 **não está incluso** no repositório. Veja [Dataset — VOC2012](#dataset--voc2012).

---

## Requisitos

- Python 3.8+
- opencv-contrib-python (`cv2.ximgproc` é necessário)
- numpy < 2
- scikit-learn
- matplotlib

---

## Instalação e Execução

### 1. Clone o repositório

```bash
git clone https://github.com/mauricio1804/CBIR_Recuperacao_de_imagem.git
cd CBIR_Recuperacao_de_imagem
```

### 2. Crie e ative o ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### 3. Prepare o dataset

Baixe o [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) e extraia na raiz do projeto. A estrutura esperada é:

```
VOCtrainval_11-May-2012/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
            └── Main/
                └── trainval.txt
```

### 4. Execute

```bash
# sem ambiente virtual no seu projeto local
python3 main.py

# com ambiente virtual no seu projeto local
source .venv/bin/activate
python3 main.py
```

Os rankings serão salvos automaticamente em `resultados/`.

---

## Dataset — VOC2012

| Parâmetro | Valor |
|---|---|
| Dataset | Pascal VOC 2012 |
| Classes utilizadas | `cat`, `horse` |
| Imagens indexadas | até 100 |
| Queries selecionadas | 5 (mais conectadas semanticamente) |
| Seleção reprodutível | `random.seed(42)` |

As queries são escolhidas priorizando as imagens cujas classes aparecem com maior frequência no restante do índice, garantindo que existam muitos documentos relevantes para uma avaliação informativa.

---

## Hiperparâmetros

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| `MAX_DOCS` | 100 | Máximo de imagens no índice |
| `MAX_QUERIES` | 5 | Número de queries avaliadas |
| `NUM_WORDS` | 300 | Palavras visuais no codebook K-means |
| `TOP_K` | 5 | Resultados retornados por query |
| `ALPHA` | 0.8 | Peso da similaridade visual vs. espacial |

O valor `ALPHA = 0.8` reflete a decisão de que descritores visuais (SIFT + BoVW) carregam mais informação discriminativa sobre o conteúdo semântico do que a posição relativa do objeto. O componente espacial (IoU, peso 0.2) atua como fator de desempate.

---

## Métricas de Avaliação

O sistema calcula três métricas padrão de recuperação de informação:

| Métrica | Descrição |
|---|---|
| **MAP@K** | Mean Average Precision — avalia a qualidade do ranqueamento |
| **Precision@K** | Proporção de resultados relevantes no top-K |
| **Recall@K** | Proporção de documentos relevantes recuperados dentre o total |

A relevância é determinada por interseção de classes entre a query e o candidato (ground truth aproximado via anotações VOC).

---

## Resultados

Resultados obtidos sobre as 5 queries selecionadas (todas da classe `cat`):

| Query | Labels | AP | P@5 | R@5 |
|---|---|---|---|---|
| 2011_000789.jpg | cat | 1.000 | 1.000 | 0.074 |
| 2008_005716.jpg | cat | 0.333 | 0.400 | 0.029 |
| 2008_005300.jpg | cat | 1.000 | 1.000 | 0.074 |
| 2009_003663.jpg | cat | 1.000 | 1.000 | 0.074 |
| 2010_004963.jpg | cat | 0.760 | 0.800 | 0.059 |

| Métrica Final | Valor |
|---|---|
| **MAP@5** | **0.8187** |
| **Precision@5** | **0.8400** |
| **Recall@5** | **0.0618** |

O Recall baixo é esperado dado o número reduzido de resultados retornados (TOP_K=5) frente ao total de imagens relevantes no índice (≈68 por query).

---

## Limitações

- **Confusão entre classes:** como `cat` e `horse` compartilham texturas similares (pelos, bordas), o sistema pode retornar imagens de cavalo em queries de gato e vice-versa.
- **Recall baixo:** com apenas TOP_K=5 resultados e ~68 documentos relevantes por query, o recall máximo teórico é de ~7,4%.
- **Ground truth aproximado:** a relevância é calculada por interseção de classes, sem pareamento objeto-a-objeto exato entre instâncias da query e da candidata.
- **Sem deep features:** a abordagem BoVW + SIFT é mais limitada semanticamente comparada a features de redes neurais convolucionais.

---

## Referências

- [Pascal VOC 2012 Challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- Lowe, D. G. (2004). *Distinctive image features from scale-invariant keypoints*. IJCV.
- Sivic, J. & Zisserman, A. (2003). *Video Google: A text retrieval approach to object matching in videos*. ICCV.
