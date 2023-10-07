# Quantization LLM

### SmoothQuant

- 解决的问题

  - 不同于CNN模型或者小型的Transformer模型，LLM模型的矩阵乘法产生的Activations 有较多的outliers, 增加量化难度。 

  - outliers 分布： 通常集中于个别的channel 维度

    | ![image-20230517144642046](C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230517144642046.png) | <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230517144753627.png" alt="image-20230517144753627" style="zoom: 50%;" /> |
    | ------------------------------------------------------------ | ------------------------------------------------------------ |

    这导致一个问题，传统量化组合有，

    - (a) Per-tensor 量化： AQ per-tensor,  WQ **per-tensor**
    - (b) Per-Channel 量化： AQ per-tensor, WQ **per-channel**

    而为了提高LLM activation outliers 特有的问题，牺牲运算速度而提出的(c)更细粒度的方案可能不没有比（b）获得太大的进步。

    - (c) AQ per-token, WQ per-channel 

    <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230517150232355.png" alt="image-20230517150232355" style="zoom:50%;" />

- 算法本质

  - 缩放张量的范围, 从而迁移量化难度。 缩Aactivations, 放Weights:  AQ hard + WQ very easy  ->  AQ easy +WQ easy 

  ![image-20230517143927051](C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230517143927051.png)

- 算法内容

  - 核心： 根据outliers 计算 scale diag, 并缩放张量
    $$
    Y= (X\cdot diag(s)^{-1}) \times (diag(s)\cdot W) = \hat{X}\hat{W}
    $$
    其中 $X\in \mathbb{R}^{T\times C_i}$ , $W \in \mathbb{R}^{C_i\times C_o}$,  $diag(s) \in \mathbb{R}^{C_i\times C_i}$. 

    ![image-20230517150426042](C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230517150426042.png)

  - s 的计算
    $$
    s_j = \max(|X_j|)^{\alpha} / \max(|W_j|)^{1-\alpha}
    $$
       when $\alpha=0.5$,  
    $$
    s_j =\sqrt{ \max(|X_j|)/ \max(|W_j|) }
    $$
    ![image-20230517151543930](C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230517151543930.png)

  - MatMul  case: 即两个Activations

    -  $Y= (X_1\cdot diag(s)^{-1}) \times (diag(s)\cdot X_2) = \hat{X_1}\hat{X_2}$
    -  s 按同理计算

  - 实际计算

    calibration set:  ${X_i}, i = 1, ..., N$,  $N$ 为样本数， 计算得到的$s$取均值, $s=1/\sum_n{s_i}$ 



### GPTQ

- 解决的问题

  - 仅权重量化 :    $argmin_{\hat{W}} \parallel WX - \hat{W}X \parallel_2^2$。 

  - Hessian矩阵指的即是该二次函数的Hessian:  $H_F=2X_F X_F^T$ ，而非反向传播中的Hessian   

- 过程细节

  - $W\in \mathbb{R}^{K\times M}$,  $X \in \mathbb{R}^{M \times N}$, $Y = W \times X \in \mathbb{R}^{K \times N}$ 

  - 逐层调试，基于calibrate set 的Hessian:   $H= \frac{1}{N} \Sigma_{i=1}^{N} {2X X^T + \lambda I} \in \mathbb{R} ^ {M\times M}$ , 取均值，加扰动项确保可逆。

  - act order sort（大值的column先做量化，小值的update后做量化）:  基于 $diag(H)$ 对 $W$ 基于M维度作列重排（降序）, $H$ 同理在两个维度上重排 

    <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518144836329.png" alt="image-20230518144836329" style="zoom: 50%;" />

    

  - Cholesky 分解求逆：$u =cholesky(H)$,  $H^{-1} = cholesky\_inverse(u)$. 

  - 以下基于 $H^{-1}$ 对 $W$ 作量化， $B=128$  分块量化，逐块量化

    <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518161253591.png" alt="image-20230518161253591" style="zoom: 67%;" />

  - （inner loop）针对每个block，作逐列量化，计算误差，并对未量化的columns基于误差更新

    <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518161451986.png" alt="image-20230518161451986" style="zoom:67%;" />

    

    <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518161657875.png" alt="image-20230518161657875" style="zoom:67%;" />

  - (outer loop) 更新该block 后面的权重

    $W_{:, (i+B):} \leftarrow  W_{:, (i+B):}-E \times H_{i:(i:B),i:(i+B)}^{-1}$

    <img src="C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518162201990.png" alt="image-20230518162201990" style="zoom:67%;" />

  - Group size 参数 (TODO) 

  - Process of pseudo  code 

![image-20230518170823539](C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518170823539.png)

### ZeroQuant

- 概述

  - AQ/WQ 全量化方案
  - LKD (Layer Knowledge Distillation) 原理上和AdaQuant 相同

- 细节

  - AQ: per-token 量化，不同于常规的per-tensor量化，旨在提供更好的精度

  - WQ: per-group 量化，应是介于per-tensor 与 per-channel 之间的量化，且对group的选择基于 GPU的结构考虑，hardware-friendly

  - Per-layer KD

    原理与AdaQuant 中累积误差的Sequential-mode 相同

    ![image-20230518174238408](C:\Users\Jon\AppData\Roaming\Typora\typora-user-images\image-20230518174238408.png)

  - 

​	

### Reference

1.  SmoothQuant

   2023, SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

2.  GPTQ

   2023, ICLR, GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. 

3. GPTQ 介绍

   https://zhuanlan.zhihu.com/p/616969812

4. ZeroQuant

   2022, ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers

5. ZeroQuant in microsoft/deepspeed

   https://github.com/microsoft/DeepSpeed/blob/master/docs/_tutorials/model-compression.md#2-tutorial-for-zeroquant-efficient-and-affordable-post-training-quantization

   

