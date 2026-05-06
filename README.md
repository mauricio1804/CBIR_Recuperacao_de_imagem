CBIR (Content-Based Image Retrieval)

Como usar

1. Crie e ative um ambiente virtual (recomendado):

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

2. Rode o script:

```bash
# com venv ativado
python3 main.py

# ou sem ativar manualmente
./.venv/bin/python3 main.py

# ou usando o helper
./run.sh
```

Observações

- O projeto usa `opencv-contrib-python` (Selective Search está em `cv2.ximgproc`).
- O `matplotlib` é configurado para modo 'Agg' automaticamente quando não há `DISPLAY`, portanto `visualize_results` salva figuras sem abrir janelas.
- Se ocorrerem erros relacionados a NumPy (módulos do sistema compilados para NumPy 2.x), use o ambiente virtual fornecido com `requirements.txt` (contém `numpy<2`).

Metodologia implementada

- Pré-processamento com propostas de regiões por `Selective Search` (e opção de `Sliding Window`).
- Extração de descritores locais via SIFT.
- Indexação por BoVW com codebook gerado por K-means.
- Ranking combinando similaridade visual (cosseno) e similaridade espacial normalizada (posição e tamanho relativos da região) com peso `alpha`.
- Avaliação com mAP e métrica de localização `Localization@5 (IoU >= 0.5)`.

Limitações da avaliação

- O ground truth de relevância para mAP é aproximado por interseção de classes dos objetos do VOC entre query e imagem candidata.
- A métrica `Localization@5` é query-aware por classe compartilhada e verifica IoU entre propostas e ground truth da imagem candidata; ainda não realiza pareamento objeto-a-objeto exato entre instâncias da query e da candidata.
