# 0716325 CV HW1

## Implementation & Enhance Methods

### Normal Estimation
在這部分，首先我將6張圖片使用提供的`read_bmp()`讀取，並將每張圖片攤平為一維向量，並將這6張相片堆疊為2為的I矩陣。  
接著讀入LightSource.txt中的光線向量，並對其做Normalization。在完成後，使用Pseudo Inverse去計算KdN  

/$$ K_dN = (L^TL)^{-1} * L^TI $$

得到KdN之後我對其做Normalization便得到了這張影像的Normal Map  
![](https://i.imgur.com/6NIPB1r.png)

### Surface Reconstruction 1 (Integration Method)

在這一部分，我先是按照課本的方法，先對x方向做積分，然後再對y方向做積分。然而這樣子會導致重建出來的影像產生許多不連續的條紋，如下圖。這是由於圖上大部分的點都只考慮到在y方向的積分訊息，而未去考慮到在x方向上相鄰點的梯度資訊。  
![](https://i.imgur.com/f55VjFB.png)  
於是，我將方法改進，除了 row 0 與 column 0 都只有對一個方向積分。此外其他的點，我都分別使用他們上方以及左方的像素點進行積分並取平均，並得到下圖的結果。可以看出像素點之間平滑了不少，然而對於距離座標(0,0)太遠的點，累積誤差仍然十分巨大，例如下圖中兔腳的部分，距離平面的高度落差太大。  
![](https://i.imgur.com/ZgyjVbv.png)  
最後，我分別從圖片的4個頂點去進行積分，並將得到的4張深度圖取平均值得到下圖。可以看到效果是十分的好，幾乎看不出有非常不連續的像素點  
![](https://i.imgur.com/JGBmlPe.png)  

### Surface Reconstruction 2 (Linear System Method)
在這部分，我首先按照投影片上的方式去建造M矩陣以及V矩陣。然而M矩陣的大小太大了，使用numpy建立時會無法配置如此大的記憶體。於是我改使用 scipy 的 sparse metrix 以節省記憶體空間。而V矩陣就一樣維持使用 numpy array 即可。  
接下來便是求解Mz=V這個方程組，我一開始使用了 Psuedo Inverse 去求解，然而使用 Psuedo Inverse 會需要去計算一個巨大矩陣的 inverse ，時間複雜度過高。於是我改為使用`scipy.sparse.linalg.lsmr()`來進行求解，他使用了迭代法來求解，可以獲得一個近似的最小平方解。得到的結果如下圖，與 Integration Method 的結果相當的接近  
![](https://i.imgur.com/OmA1SHR.png)  




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

