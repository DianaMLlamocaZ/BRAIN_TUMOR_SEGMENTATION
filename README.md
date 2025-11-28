# Brain Tumor Segmentation
## Descripción
- En este repositorio, almaceno un proyecto de segmentación binaria de tumores cerebrales, implementando una U-Net *from scratch*.

- *Dataset:* El dataset utilizado fue "Brain Tumor Image DataSet", disponible en Kaggle. 

====

## Código
### - Load Data:
- **custom_dataset.py:**
  - Archivo que contiene el código para la creación del custom dataset.
  - **Objetivo:** Para evitar cargar todas las imágenes y máscaras en memoria, se creó el custom dataset con la finalidad de obtener los datos solo en el momento de su utilización --> Lazy load.
  - **Nota:** Incluye el preprocesamiento de las imágenes y máscaras (se puede aplicar Data Augmentation)
  
- **read_img_json:**
  - Lee el archivo json (donde se encuentran las anotaciones) y retorna el nombre de la imagen más el *bounding box* (contorno) de la máscara de segmentación.
  - **Objetivo:** Función útil para permitir el *lazy load* en el custom dataset.


### - Modelos:
- **modelo1.py:** UNet implementada desde cero, con pocos parámetros (menor a 1 millón)
  - Modelo pequeño de prueba para verificar el *load* correcto de los datos.
    
- **modelo2.py:** UNet implementada desde cero, con más parámetros (mayor a 4 millones)
  - UNet implementada desde cero: DownSampling, Bottleneck, UpSampling.
  - Adicional, contiene la operación *crop* para permitir la concatenación los feature maps generados por el encoder (mayor tamaño) con los creados por el decoder (menor tamaño). Y, así, permitir las *concatenation skip connections*.

### - Training:
- **training.py:**
  - Archivo que contiene el código para el entrenamiento del modelo. 


### - Evaluación:
- **metrics.py:**
  - Contienen las **loss functions** y métricas a utilizar para entrenar y evaluar el **performance** del modelo.
  - **Objetivo:**: Facilitar el uso directo de loss functions y métricas durante el entrenamiento y evaluación.
