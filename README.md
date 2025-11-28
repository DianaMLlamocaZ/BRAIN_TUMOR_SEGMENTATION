# Brain Tumor Segmentation
## Descripción
- En este repositorio, almaceno un proyecto de segmentación binaria de tumores cerebrales, implementando una U-Net *from scratch*.

- *Dataset:* El dataset utilizado fue "Brain Tumor Image DataSet", disponible en Kaggle. 

====

## Código
- **custom_dataset.py:**
  - Archivo que contiene el código para la creación del custom dataset.
  - Objetivo: Para evitar cargar todas las imágenes y máscaras en memoria, se creó el custom dataset con la finalidad de obtener imágenes en el momento de su utilización --> Lazy load.
  
