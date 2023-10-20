# 3Dreconstruction
## 简介
参考[astra-cbct](https://github.com/katherinekin/virtual-cbct)进行CBCT重建，重点解决planar的问题

## 框架
1. 原始cbct重建demo：生成phantom，生成投影，进程重建
   
phantom.py —— 预先设定的长方体模型

astra_cone.py —— 3d geometry设置为cone的实现（只支持xray，target，detector在一个平面的情况），结果在output中

astra_cone_vec.py —— 3d geometry设置为cone_vec的实现，对预置长方体模型的重建，可以自定义投影矩阵（12*1），结果在output1中


3. 对实验数据的重建
astra_cone_vec2.py —— 根据具体数据设置投影矩阵重建，结果在output3中

## 进展
1. 对预置长方体模型，可以实现cone和cone_vec的投影和重建
2. astra_cone_vec2.py中进行了各种参数遍历：
   
   —— 目前FDK_CUDA方法重建slice大部分地方为0，只有少数亮点
   ![image](https://github.com/xjs0227/3Dreconstruction/blob/main/imgs/FDK.jpg)
   
   —— SIRT3D_CUDA方法重建slice出现圆环状，意义不明
   
   ![image](https://github.com/xjs0227/3Dreconstruction/blob/main/imgs/SIRT.png)
   
4. astra_cone.py中对预置长方体模型进行不同角度的投影生成，期望找到契合实验数据的投影矩阵参数
