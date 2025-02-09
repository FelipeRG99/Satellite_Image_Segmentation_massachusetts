import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm 

def from_tiff_to_png(data_path_images,data_path_end,type_='img'):
    """
    FUNCTION: Convert tiff/tif files to png

    PARAMS: 
        data_path_images: source path
        data_path_end: destination path
        type_: indicates if the files are regular images or masks
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
def obtain_cuts(img,cut=224,tipo='img'):
    """
    FUNCTION: Obtain slices of a given image of a specified size (cut x cut)

    PARAMS: 
        img: np.array of the image loaded by cv2.imread()
        cut: size of each slice
        type: indicates if it is an image or its mask (msk)

    RETURN:
        array_tot: np.array(number_of_slices, cut, cut, channels) where channels are 3 for regular images
        and none for masks

    """
    # Get the dimensions of the image (height, width, channels)

    if tipo == 'img':
        height, width, _ = img.shape
        formato=(1,cut,cut,3)
    elif tipo == 'msk':
        height, width= img.shape
        formato=(1,cut,cut)

    #coordinates for the cuts in the loop
    max_val_cut=height-cut
    max_val_cut_x=width-cut
    array_tot=np.array([])
    #loop in X axis
    for j in range(1,int(width/cut)+2):
        if cut*j<width:
            cut_ini_x=cut*(j-1)
            cut_fin_x=cut*j
        else:
            cut_ini_x=max_val_cut_x
            cut_fin_x=width
        #loop in Y axis
        for i in range(1,int(height/cut)+2):
            if cut*i<height:
                cut_ini=cut*(i-1)
                cut_fin=cut*i
            else:
                cut_ini=max_val_cut
                cut_fin=height
            # do the cuts on the img
            if tipo == 'img':
                img_cut = img[cut_ini:cut_fin, cut_ini_x:cut_fin_x,:]
            elif tipo == 'msk':
                img_cut = img[cut_ini:cut_fin, cut_ini_x:cut_fin_x]
            #group the img cuts in an array
            if array_tot.shape[0]>0:
                array_tot=np.vstack([array_tot,img_cut.reshape(formato)])
            else:
                array_tot=img_cut.reshape(formato)
    return array_tot
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
def reconstruct_img(img_cuts,original_height,original_width,type_='img'):
    """
    FUNCTION: Reconstruct the original image from its slices

    PARAMS: 
        img_cuts: np.array(num, cut, cut) array with the image slices
        original_height, original_width: int size of the original image height and width
        type_: str type of image, either regular image or mask
    RETURN:
        reconstructed_img: np.array(original_shape, original_shape) array of the final reconstructed image

    """
    num_cuts=img_cuts.shape[0]
    cut=img_cuts.shape[1]
    num_cuts_axis_x=(original_width//cut)+1
    num_cuts_axis_y=(original_height//cut)+1

    #create base array
    reconstructed_img=np.array([])
    if type_ == 'img':
        reconstructed_img=np.zeros((original_height,original_width,3))
    elif type_=='msk':
        if len(img_cuts.shape)==3:
            reconstructed_img=np.zeros((original_height,original_width))
        elif len(img_cuts.shape)==4:
            reconstructed_img=np.zeros((original_height,original_width,1))
    index_tot=0
    #loop in X axis
    for i in range(num_cuts_axis_x):
        if i!=num_cuts_axis_x-1:
            corte_ini_x=(i)*cut
            corte_fin_x=(i+1)*cut
        else:
            corte_ini_x=original_width-cut
            corte_fin_x=original_width
        #loop in Y axis
        for j in range(num_cuts_axis_y):
            if j!=num_cuts_axis_y-1:
                corte_ini_y=(j)*cut
                corte_fin_y=(j+1)*cut
            else:
                corte_ini_y=original_height-cut
                corte_fin_y=original_height
            #add cut to final reconstructed array
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
    FUNCTION: Given directories containing images and masks, obtain the slices of each

    PARAMS:
        data_path_images, data_path_masks: paths of images and masks
        cut: size of each slice
        positive_min_mask: indicates the minimum factor of white points/positive values in a mask. 
            This is ideal for filtering images with a lot of background (negative values) vs. information 
            (mask/positive values)
        limit: limits the number of images to be processed. Ideal if we have too many images
    RETURN:
        white_factor: float white pixels / total pixels
        final_images: np.array(num, cut, cut, 3) array with the resulting slices for images
        final_masks: np.array(num, cut, cut) array with the resulting slices for masks
    """

    data_list_images = os.listdir(data_path_images)
    final_images = []
    final_masks = []
    num_img_discard=0
    mask_factors=[]

    limite = len(data_list_images) if limite == 0 else limite

    for file_name in tqdm(data_list_images[:limite]):
        # read img
        img = cv2.imread(os.path.join(data_path_images, file_name))[:,:,:3]
        mask_file = os.path.splitext(file_name)[0] + '.png'  # mask name
        mask_path = os.path.join(data_path_masks, mask_file)
        if os.path.exists(mask_path):
            # read mask
            mask = cv2.imread(mask_path)[:,:,1]
            mask_factor=num_percentage(mask,255)
            if mask_factor>positive_min_mask:
                mask_factors.append(mask_factor)
                # obtain cuts
                img_cuts=obtain_cuts(img,cut,tipo='img')
                #add cuts to list
                img_trozos_list=[trozo for trozo in img_cuts]
                final_images+=img_trozos_list
                #add cuts from mask
                mask_cuts=obtain_cuts(mask,cut,tipo='msk')
                #add cuts to list
                mask_trozos_list=[trozo for trozo in mask_cuts]
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
    FUNCTION: Returns the ratio of pixels with a specific value to the total pixels in the image

    PARAMS:
        img: np.array of the image loaded by cv2.imread()

    RETURN:
        num_factor: float ratio of pixels with the specific value to total pixels

    """

    total_pixels = img.shape[0] * img.shape[1]
    num_pixels = np.sum(img == num)  # sum pixels
    num_factor=num_pixels / total_pixels
    return num_factor
#####################################################################################################################
##################################                                                  #################################
#####################################################################################################################
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