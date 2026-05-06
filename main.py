import os
import random
import xml.etree.ElementTree as elementtree
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

VOC_CLASSES = {'cat', 'horse'}

# ── Dataset ───────────────────────────────────────────────────────────────────

def parse_annotation(xml_path):
    root = elementtree.parse(xml_path).getroot()
    objects = []
    for obj in root.findall('object'):
        label = obj.findtext('name')
        if label not in VOC_CLASSES:
            continue
        b = obj.find('bndbox')
        objects.append({
            'label': label,
            'bbox': [int(float(b.findtext(t))) for t in ('xmin', 'ymin', 'xmax', 'ymax')],
        })
    return objects


def load_dataset(voc_root, max_docs, max_queries):
    imgs_dir = os.path.join(voc_root, 'JPEGImages')
    ann_dir = os.path.join(voc_root, 'Annotations')
    ids = open(os.path.join(voc_root, 'ImageSets',
               'Main', 'trainval.txt')).read().split()
    random.shuffle(ids)

    docs = []
    for img_id in ids:
        img_path = os.path.join(imgs_dir, f'{img_id}.jpg')
        ann_path = os.path.join(ann_dir,  f'{img_id}.xml')
        if not (os.path.isfile(img_path) and os.path.isfile(ann_path)):
            continue
        objects = parse_annotation(ann_path)
        if not objects:
            continue
        labels = sorted({o['label'] for o in objects})
        docs.append({'image_path': img_path,
                    'objects': objects, 'labels': labels})
        if len(docs) >= max_docs:
            break

    label_map = {d['image_path']: set(d['labels']) for d in docs}
    candidates = sorted(
        [(d, sum(1 for p, l in label_map.items()
                 if p != d['image_path'] and set(d['labels']) & l))
         for d in docs],
        key=lambda x: x[1], reverse=True
    )
    
    selected_queries = [d for d, n in candidates[:max_queries]]

    print(f'\nQueries selecionadas ({len(selected_queries)}):')
    for q in selected_queries:
        print(f'  • {os.path.basename(q["image_path"])} → {q["labels"]}')

    return docs, selected_queries


def object_proposals(objects):
    return [i['bbox'] for i in objects if i.get('bbox')]


def sift_descriptors(region, sift):
    gray = cv2.cvtColor(
        region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    _, des = sift.detectAndCompute(gray, None)
    return des


def build_codebook(descriptors_list, k):
    all_des = np.vstack([d for d in descriptors_list if d is not None])
    kmeans = KMeans(n_clusters=min(k, len(all_des)),
                    random_state=42, n_init=10)
    kmeans.fit(all_des)
    return kmeans


def bovw(descriptors, kmeans):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans.n_clusters)
    hist = np.bincount(kmeans.predict(descriptors),
                       minlength=kmeans.n_clusters).astype(float)
    return hist / hist.sum() if hist.sum() > 0 else hist

# ── Similaridade espacial (IoU) ───────────────────────────────────────────────

def bbox_to_spatial(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    return np.array([(x1 + bw/2)/w, (y1 + bh/2)/h, bw/w, bh/h], dtype=np.float32)


def iou_spatial(s_a, s_b):
    def to_xyxy(s):
        cx, cy, w, h = s
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2
    ax1, ay1, ax2, ay2 = to_xyxy(s_a)
    bx1, by1, bx2, by2 = to_xyxy(s_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1)*(ay2 - ay1) + (bx2 - bx1)*(by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0

# ── Indexação ─────────────────────────────────────────────────────────────────

def index_images(docs, sift, num_words, annotations):
    print('Indexando imagens...')
    image_features, all_des = {}, []

    for i, doc in enumerate(docs):
        path = doc['image_path']
        print(f'  [{i+1}/{len(docs)}] {os.path.basename(path)}')
        img = cv2.imread(path)
        if img is None:
            print(f'Não foi possível carregar {path}, pulando.')
            continue

        objs = annotations.get(path, [])
        proposals = object_proposals(objs)

        feats = []
        for bbox in proposals:
            x1, y1, x2, y2 = bbox
            region = img[y1:y2, x1:x2]
            if region.size == 0 or (y2 - y1) < 10 or (x2 - x1) < 10:
                continue
            des = sift_descriptors(region, sift)
            if des is not None:
                all_des.append(des)
                feats.append({'bbox': bbox, 'descriptors': des,
                              'spatial': bbox_to_spatial(bbox, img.shape)})
        image_features[path] = feats

    print('Construindo codebook (K-means)...')
    kmeans = build_codebook(all_des, num_words)
    for feats in image_features.values():
        for f in feats:
            f['bovw'] = bovw(f['descriptors'], kmeans)

    return image_features, kmeans

# ── Consulta ──────────────────────────────────────────────────────────────────

def run_query(query_path, image_features, sift, kmeans,
              top_k=5, alpha=0.7, query_objects=None, verbose=True):
    if verbose:
        print(f'\nConsultando: {os.path.basename(query_path)}')
    img = cv2.imread(query_path)
    if img is None:
        return []

    proposals = object_proposals(query_objects)

    q_feats = []
    for bbox in proposals:
        x1, y1, x2, y2 = bbox
        region = img[y1:y2, x1:x2]
        if region.size == 0 or (y2 - y1) < 10 or (x2 - x1) < 10:
            continue
        des = sift_descriptors(region, sift)
        if des is not None:
            q_feats.append({'bovw': bovw(des, kmeans),
                            'spatial': bbox_to_spatial(bbox, img.shape)})
            
    if not q_feats:
        return []

    results = []
    for path, feats in image_features.items():
        if path == query_path or not feats:
            continue

        best_score, best_bbox = -1.0, None
        for qf in q_feats:
            qv = qf['bovw'].reshape(1, -1)
            if np.linalg.norm(qv) == 0:
                continue
            for cf in feats:
                cv_ = cf['bovw'].reshape(1, -1)
                if np.linalg.norm(cv_) == 0:
                    continue
                vis = float(cosine_similarity(qv, cv_)[0][0])   
                spat = iou_spatial(qf['spatial'], cf['spatial'])  
                score = alpha * vis + (1.0 - alpha) * spat        
                if score > best_score:
                    best_score, best_bbox = score, cf['bbox']

        if best_score >= 0:
            results.append(
                {'path': path, 'score': best_score, 'bbox': best_bbox})

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# ── Avaliação MAP + Precision@K + Recall@K ───────────────────────────────────

def evaluate(query_docs, image_features, labels, sift, kmeans, top_k=5):
    print()
    aps, precisions, recalls = [], [], []

    for doc in query_docs:
        path = doc['image_path']
        query_label = set(doc['labels'])

        results = run_query(path, image_features, sift, kmeans,
                            top_k=top_k, query_objects=doc['objects'], verbose=False)

        n_rel, prec_sum = 0, 0.0
        for rank, r in enumerate(results, 1):
            if query_label & set(labels.get(r['path'], [])):
                n_rel += 1
                prec_sum += n_rel / rank

        total_rel = sum(1 for p, l in labels.items()
                        if p != path and query_label & set(l))
        denom = min(total_rel, top_k)

        # FIX 5 — Precision@K e Recall@K como métricas complementares ao MAP
        precision_k = n_rel / top_k if top_k > 0 else 0.0
        recall_k = n_rel / total_rel if total_rel > 0 else 0.0
        precisions.append(precision_k)
        recalls.append(recall_k)

        print(f'  {os.path.basename(path):<30} labels={sorted(query_label)}')
        print(
            f'    relevantes={total_rel}  precision_sum={prec_sum:.3f}', end='')

        if denom > 0:
            ap = prec_sum / denom
            aps.append(ap)
            print(
                f'  AP={ap:.3f}  P@{top_k}={precision_k:.3f}  R@{top_k}={recall_k:.3f}')
        else:
            print('  AP=N/A (sem relevantes no dataset)')

    map_score = float(np.mean(aps)) if aps else 0.0
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0

    print(f'\n>>> MAP{top_k}      = {map_score:.4f}')
    print(f'>>> Precision{top_k} = {mean_prec:.4f}')
    print(f'>>> Recall{top_k}    = {mean_recall:.4f}')
    return map_score, mean_prec, mean_recall

# ── Visualização ──────────────────────────────────────────────────────────────

def visualize_results(query_path, results, query_bbox=None):
    n = len(results) + 1
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    def show(ax, path, bbox, title, color='lime'):
        img = cv2.imread(path)
        if img is None:
            ax.axis('off')
            return
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        if bbox:
            x1, y1, x2, y2 = bbox
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor=color, facecolor='none'))

    show(axes[0], query_path, query_bbox,
         f'Query: {os.path.basename(query_path)}', color='deepskyblue')
    for i, r in enumerate(results, 1):
        show(axes[i], r['path'], r.get('bbox'),
             f'#{i} {os.path.basename(r["path"])}\nScore: {r["score"]:.3f}')
    for ax in axes[n:]:
        ax.axis('off')

    fig.tight_layout()
    os.makedirs('resultados', exist_ok=True)
    out = os.path.join('resultados',
                       f'ranking_{os.path.splitext(os.path.basename(query_path))[0]}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Figura salva: {out}')
    plt.show()
    plt.close(fig)

# ── Execução principal ────────────────────────────────────────────────────────

def main():
    MAX_DOCS = 100
    MAX_QUERIES = 5
    NUM_WORDS = 300
    TOP_K = 5
    ALPHA = 0.8
    random.seed(42)

    voc_root = 'VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    print(f'VOC2012: {voc_root}')

    docs, query_docs = load_dataset(voc_root, MAX_DOCS, MAX_QUERIES)
    print(f'{len(docs)} documentos | {len(query_docs)} queries\n')

    labels = {d['image_path']: d['labels'] for d in docs}
    annotations = {d['image_path']: d['objects'] for d in docs}

    sift = cv2.SIFT_create()
    image_features, kmeans = index_images(docs, sift, NUM_WORDS, annotations)

    print('\n' + '='*50 + '\nCONSULTAS\n' + '='*50)
    for doc in query_docs:
        path = doc['image_path']
        q_bbox = doc['objects'][0]['bbox'] if doc['objects'] else None
        results = run_query(path, image_features, sift, kmeans,
                            top_k=TOP_K, alpha=ALPHA, query_objects=doc['objects'])

        print(f'\n{os.path.basename(path)} {doc["labels"]}:')
        for i, r in enumerate(results):
            rl = labels.get(r['path'], [])
            ok = '✓' if set(doc['labels']) & set(rl) else '✗'
            print(
                f'  {i+1}. {ok} {os.path.basename(r["path"])} {rl} — {r["score"]:.3f}')

        visualize_results(path, results, query_bbox=q_bbox)

    print('\n' + '='*50 + '\nAVALIAÇÃO\n' + '='*50)
    evaluate(query_docs, image_features, labels, sift, kmeans, TOP_K)


if __name__ == '__main__':
    main()
