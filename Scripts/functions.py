import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm 

def from_tiff_to_png(data_path_images,data_path_end,type_='img'):
    """
    FUNCION:   Pasar de archivos tiff/tif a png

    PARAMS: 
        data_path_images:path origen
        data_path_end: path final
        type_: indica si son imagenes normales o mascaras
    RETURN:
    
    """
    data_list_images = os.listdir(data_path_images)
    for file in tqdm(data_list_images):
        image = cv2.imread(data_path_images+"\\"+file)
        if type_=='img':
            formato='tiff'
        elif type_=='msk':
            formato='tif'
        cv2.imwrite(data_path_end+"\\"+file.replace(formato,'png'), image)
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
def obtener_corte(img,cut=224,tipo='img'):
    """
    FUNCION:    Obtener cortes de una imagen dada de un tamaño (cutxcut)

    PARAMS: 
        img: np.array de la imagen por cv2.imread()
        cut: tamaño del corte
        type: indica si es una imagen o su mascara (msk)

    RETURN:
        array_tot: np.array(numero_cortes,cut,cut,canales) donde canales son 3 en las imagenes
        normales y nada en las mascaras
    
    """
    # Obtener las dimensiones de la imagen (alto, ancho, canales)
    if tipo == 'img':
        alto, ancho, _ = img.shape
        formato=(1,cut,cut,3)
    elif tipo == 'msk':
        alto, ancho= img.shape
        formato=(1,cut,cut)

    # Definir las coordenadas para el recorte
    max_val_recorte=alto-cut
    max_val_recorte_x=ancho-cut
    array_tot=np.array([])
    #bucle ene le eje X
    for j in range(1,int(ancho/cut)+2):
        if cut*j<ancho:
            corte_ini_x=cut*(j-1)
            corte_fin_x=cut*j
        else:
            corte_ini_x=max_val_recorte_x
            corte_fin_x=ancho
        #Bucle en el eje Y
        for i in range(1,int(alto/cut)+2):
            if cut*i<alto:
                corte_ini=cut*(i-1)
                corte_fin=cut*i
            else:
                corte_ini=max_val_recorte
                corte_fin=alto
            # Realizar el recorte de la imagen
            if tipo == 'img':
                trozo_imagen = img[corte_ini:corte_fin, corte_ini_x:corte_fin_x,:]
            elif tipo == 'msk':
                trozo_imagen = img[corte_ini:corte_fin, corte_ini_x:corte_fin_x]
            #agrupar imagenes en un mismo array
            if array_tot.shape[0]>0:
                array_tot=np.vstack([array_tot,trozo_imagen.reshape(formato)])
            else:
                array_tot=trozo_imagen.reshape(formato)
    return array_tot
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
def white_percentage(img):
    """
    FUNCION: Hay imagenes que tienen cortes blancos debido a mala fotografia, estas se pueden eliminar,
        para ello esta funcion calcula cuantos pixeles blancos 255 hay en una imagen

    PARAMS:
        img: np.array de la imagen por cv2.imread()

    RETURN:
        white_factor= float pixeles blancos/ pixeles totales
    
    """
    total_pixels = img.shape[0] * img.shape[1]
    white_pixels = np.sum(img == 255)  # Contar píxeles blancos
    white_factor=white_pixels / total_pixels
    return white_factor
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
def reconstruir_imagen(img_cuts,original_shape,type_='img'):
    """
    FUNCION: A partir de los cortes de una imagen, reconstruir la imagen original

    PARAMS: 
        img_cuts: np.array(num,cut,cut,) array con los cortes de la imagen
        original_shape: int tamaño de la imagen original
        type_: str tipo de imagen, imagen o mascara
    RETURN:
        reconstructed_img: np.array(original_shape,original_shape,) array de la img final reconstruida
    
    """
    num_cortes=img_cuts.shape[0]
    recorte=img_cuts.shape[1]
    num_cortes_eje=int(np.sqrt(num_cortes))
    #crear array base
    reconstructed_img=np.array([])
    if type_ == 'img':
        reconstructed_img=np.zeros((original_shape,original_shape,3))
    elif type_=='msk':
        if len(img_cuts.shape)==3:
            reconstructed_img=np.zeros((original_shape,original_shape))
        elif len(img_cuts.shape)==4:
            reconstructed_img=np.zeros((original_shape,original_shape,1))
    index_tot=0
    #bucle en el eje x
    for i in range(num_cortes_eje):
        if i!=num_cortes_eje-1:
            corte_ini_x=(i)*recorte
            corte_fin_x=(i+1)*recorte
        else:
            corte_ini_x=original_shape-recorte
            corte_fin_x=original_shape
        #bucle en el eje y
        for j in range(num_cortes_eje):
            if j!=num_cortes_eje-1:
                corte_ini_y=(j)*recorte
                corte_fin_y=(j+1)*recorte
            else:
                corte_ini_y=original_shape-recorte
                corte_fin_y=original_shape
            #añadir recorte al array final
            if type_ == 'img':
                reconstructed_img[corte_ini_y:corte_fin_y,corte_ini_x:corte_fin_x,:]=img_cuts[index_tot]
            elif type_=='msk':
                if len(img_cuts.shape)==3:
                    reconstructed_img[corte_ini_y:corte_fin_y,corte_ini_x:corte_fin_x]=img_cuts[index_tot]
                if len(img_cuts.shape)==4:
                    reconstructed_img[corte_ini_y:corte_fin_y,corte_ini_x:corte_fin_x,:]=img_cuts[index_tot]
            index_tot+=1
    return reconstructed_img
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
def filter_cut_images(data_path_images, data_path_masks,cut=224,positive_min_mask=0.001,limite=0):
    """
    FUNCION: Dados los directorios donde se encuentran imagenes y mascaras, obtener los recortes de cada una de ellas

    PARAMS:
        data_path_images,data_path_masks: path de imagenes y mascaras
        cut: tamaño del corte
        positive_min_mask: indica el factor minimo de puntos blancos/valores positivos en una mascara. Esto es ideal
            para filtrar imagenes con mucho fondo (valores negativos) vs información (mascara/valores positvos)
        limite: limita el numero de imagenes que se van a procesar. Ideal por si tenemos demasiadas
    RETURN:
        white_factor= float pixeles blancos/ pixeles totales
        final_images: np.array(num,cut,cut,3) array con los resultados de los cortes para las imagenes
        final_masks: np.array(num,cut,cut) array con los resultados de los cortes para las mascaras
    """

    data_list_images = os.listdir(data_path_images)
    final_images = []
    final_masks = []
    num_img_discard=0
    mask_factors=[]

    limite = len(data_list_images) if limite == 0 else limite

    for file_name in tqdm(data_list_images[:limite]):
        # Leer la imagen
        img = cv2.imread(os.path.join(data_path_images, file_name))[:,:,:3]
        mask_file = os.path.splitext(file_name)[0] + '.png'  # Nombre de la máscara
        mask_path = os.path.join(data_path_masks, mask_file)
        if os.path.exists(mask_path):
            # Leer mascara
            mask = cv2.imread(mask_path)[:,:,1]
            mask_factor=num_percentage(mask,255)
            if mask_factor>positive_min_mask:
                mask_factors.append(mask_factor)
                # Obtener cortes
                img_trozos_imagen=obtener_corte(img,cut,tipo='img')
                #agregar a lista
                img_trozos_list=[trozo for trozo in img_trozos_imagen]
                final_images+=img_trozos_list
                #obtener cortes de la mascara
                mask_trozos_imagen=obtener_corte(mask,cut,tipo='msk')
                #agregar a lista
                mask_trozos_list=[trozo for trozo in mask_trozos_imagen]
                final_masks+=mask_trozos_list
            else:
                num_img_discard+=1
        else:
            num_img_discard+=1
    print(f'Se han descartado {num_img_discard} imagenes')
    # Convertir las listas en arrays NumPy
    final_images = np.asarray(final_images, dtype=np.uint8)
    final_masks = np.asarray(final_masks, dtype=np.uint8)
    #Pasamos valores de 255 a 1
    final_masks[final_masks==255]=1
    mask_factors =np.array(mask_factors)

    return final_images, final_masks, mask_factors
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
def num_percentage(img,num=0):
    """
    FUNCION: Devuelve el factor existente entre pixeles de un numero concreto y los pixeles totales de la imagen

    PARAMS:
        img: np.array de la imagen por cv2.imread()

    RETURN:
        num_factor= float pixeles del numero num/ pixeles totales

    """
    total_pixels = img.shape[0] * img.shape[1]
    num_pixels = np.sum(img == num)  # Contar píxeles blancos
    num_factor=num_pixels / total_pixels
    return num_factor

def plot_images_comparation(list_images_rows,titles_rows=["Input Image", "Real Mask", "Predicted Mask"],save=False,filename=''):
    num_rows = len(list_images_rows)
    num_cols=len(list_images_rows[0])
    if num_cols==2:
        titles_rows=["Input Image", "Predicted Mask"]
    fig=plt.figure(figsize=(15, 5*num_rows))
    index=1
    for i in range(num_rows):
        for j in range(num_cols):
            plt.subplot(num_rows,num_cols,index)
            plt.imshow(list_images_rows[i][j])
            plt.axis('off')
            if i==0:
                plt.title(titles_rows[j])  
            index+=1  
    plt.show()
    if save and filename!='':
        fig=plt.figure(figsize=(15, 5*num_rows))
        index=1
        for i in range(num_rows):
            for j in range(num_cols):
                plt.subplot(num_rows,num_cols,index)
                plt.imshow(list_images_rows[i][j])
                plt.axis('off')
                if i==0:
                    plt.title(titles_rows[j])  
                index+=1  
        plt.savefig(filename, dpi=300, bbox_inches='tight')




print('Funciones cargadas')