#Archivo
import json

#Retorna el filename y el bbox (convertiré esa región en máscara) en el custom dataset
def read_img_json(path_dir,id_img):
    path_json=f"{path_dir}/_annotations.coco.json"

    with open(path_json,"rb") as file:
        results=json.load(file)
        
        img_filename=results["images"][id_img]["file_name"]
        annotation_img=results["annotations"][id_img]

        return img_filename, annotation_img["bbox"]
