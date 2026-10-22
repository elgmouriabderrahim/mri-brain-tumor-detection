# mri-brain-tumor-detection
Deep learning prototype for brain MRI tumor classification using Transfer Learning and tumor region detection using YOLOv8, with Airflow, Streamlit, and Docker integration.

## Étapes 1 et 2 — données

Les images sont rangées par partition et par classe dans `data/{Train,Val}/<classe>/images`. Les annotations correspondantes sont des fichiers texte dans `labels/`. Le format contrôlé est le format YOLO : `class_id x_center y_center width height`, les quatre coordonnées étant normalisées entre 0 et 1.

### 1. Explorer, contrôler et séparer

Depuis la racine du dépôt, exécuter :

```bash
python src/prepare_data.py
```

Le script vérifie l’ouverture des images, l’association image/annotation, les boîtes YOLO et la cohérence entre l’id de classe et le dossier (`0` Glioma, `1` Meningioma, `2` No Tumor, `3` Pituitary). Il calcule les empreintes SHA-256 pour repérer les copies identiques, puis génère `data/prepared/report.md`, `issues.csv`, `splits.csv` et `sample_boxes.jpg`. Le manifeste garde les copies identiques dans la même partition. Le jeu Val fourni est conservé; les groupes restants du jeu Train sont séparés en train et test (85/15 par classe, graine 42). Il ne déplace ni ne modifie les données sources. Les lignes `usable=no` doivent être corrigées ou ignorées pendant l’entraînement.

### 2. Prétraiter et augmenter

`src/data_pipeline.py` charge seulement les entrées `usable=yes` du manifeste. Il convertit les images en RGB, les redimensionne par défaut à 224×224 et transforme les pixels en nombres entre 0 et 1. Les annotations restent en coordonnées YOLO normalisées.

Les augmentations sont faites à la volée et seulement lorsque `augment=True` est demandé pour un échantillon de train : rotation aléatoire jusqu’à ±10°, translation jusqu’à 5 %, zoom de 0,90 à 1,10 et contraste entre 0,90 et 1,10. Les quatre coins de chaque boîte sont transformés avec la même rotation, translation et le même zoom que l’image; les boîtes sont recadrées aux limites de l’image. Les boîtes entièrement déplacées hors champ sont retirées. La validation et le test restent déterministes et ne sont jamais augmentés.

Pour créer une planche qui compare les images d’origine et augmentées :

```bash
python src/preview_augmentation.py
```

Le résultat est `data/prepared/augmentation_preview.jpg`. L’augmentation est une fonction Python réutilisable par les prochains scripts d’entraînement; elle ne crée pas de fichiers d’entraînement supplémentaires sur le disque.
