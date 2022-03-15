# 0716325 CV HW1

## Implementation & Enhance Methods

### Normal Estimation
在這部分，首先我將6張圖片使用提供的`read_bmp()`讀取，並將每張圖片攤平為一維向量，並將這6張相片堆疊為2為的I矩陣  
接著讀入LightSource.txt中的光線向量，並對其做Normalization。在完成後，使用Pseudo Inverse去計算$ K_dN $  
$$ K_dN = (L^TL)^{-1} * L^TI $$

得到KdN之後我對其做Normalization便得到了這張影像的Normal Map


### Surface Reconstruction 1 (Integration Method)


### Surface Reconstruction 2 (Linear System Method)


## Result

### Normal Maps

| bunny                                | star                                 | venus                                |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![](https://i.imgur.com/zyzYAjn.png) | ![](https://i.imgur.com/BcKc0pf.png) | ![](https://i.imgur.com/vlXdHRe.png) |


### Depth Maps

|       | Integration Method                   | Linear System Method                 |
| ----- | ------------------------------------ | ------------------------------------ |
| bunny | ![](https://i.imgur.com/yxTuj4I.png) | ![](https://i.imgur.com/kREt6Ho.png) |
| star  | ![](https://i.imgur.com/VDny8dt.png) | ![](https://i.imgur.com/fGmCwW8.png) |
| venus | ![](https://i.imgur.com/ikLffW3.png) | ![](https://i.imgur.com/tBfMSuT.png) |

