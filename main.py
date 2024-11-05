import os
import warnings
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score,classification_report

nueva_ruta = 'your rute'
os.chdir(nueva_ruta)

warnings.filterwarnings("ignore")
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from Scripts.functions import reconstruir_imagen,obtener_corte,plot_images_comparation

data_path_images='main_img/img'
data_path_msk_road='main_img/mask_road'
data_path_msk_build='main_img/mask_build'
file=os.listdir(data_path_images)[0]

img=cv2.imread(data_path_images+'/'+file)[:,:,:3]
try:
    mask_road=cv2.imread(data_path_msk_road+'/'+file)[:,:,1]
    mask_build=cv2.imread(data_path_msk_build+'/'+file)[:,:,1]
    img_mask=(2*(mask_road/255)+mask_build/255)
    img_mask[img_mask==3]=1
    plot_mask=True
except:
    plot_mask=False
#realizar corte
trozos_imagen=obtener_corte(img)

model_type='uni'
save=False
########################### UNET MODEL ROAD/BUILD SEP   #####################
if model_type=='sep':
    BACKBONE = 'resnet50'
    model_road = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    model_road.load_weights('Trained_models/road_models/modelo-08-0.98.weights.h5')

    model_build = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    model_build.load_weights('Trained_models/build_models/modelo-05-0.93.weights.h5')

    #predicciones
    preds_road=model_road.predict(trozos_imagen)
    pred_binary_road = (preds_road > 0.5).astype(np.uint8)*2

    preds_build=model_build.predict(trozos_imagen)
    pred_binary_build = (preds_build > 0.5).astype(np.uint8)

    pred_final=pred_binary_road+pred_binary_build
    pred_final[pred_final==3]=1
########################### UNET MODEL ROAD/BUILD UNI   #####################
elif  model_type=='uni':
    BACKBONE = 'resnet50'
    model_uni = sm.Unet(BACKBONE, classes=3, activation='softmax')
    model_uni.load_weights('Trained_models/road_build_models/modelo-07-0.87.weights.h5')    

    #predicciones
    preds_uni=model_uni.predict(trozos_imagen)
    pred_binary = (preds_uni > 0.5).astype(np.uint8)

    pred_final=pred_binary[:,:,:,1]+pred_binary[:,:,:,2]*2
    pred_final[pred_final==3]=1
#reconstruyo la prediccion
pred_mask_recon=reconstruir_imagen(pred_final,1500,type_='msk')
if plot_mask:
    # Mostrar las imágenes en filas de tres
    plot_images_comparation([[img,img_mask,pred_mask_recon]],save=save,filename='Imagenes resultados/'+model_type+'_'+file+'.png')
    # Aplanar ambos arrays para calcular la accuracy pixel a pixel
    y_test_flat = img_mask.flatten()
    pred_flat = pred_mask_recon.flatten()
    # Calcular la accuracy
    accuracy = accuracy_score(y_test_flat, pred_flat)
    print('-'*60)
    print('Accuracy\n',accuracy)
    report = classification_report(y_test_flat, pred_flat)
    print('Clasification Report\n',report)
else:
    plot_images_comparation([[img,pred_mask_recon]])


