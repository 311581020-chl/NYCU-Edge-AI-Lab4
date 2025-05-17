# Lab 4: Quantize DeiT-S

## Part 2: Analysis

### 1. A visual analysis of the weight/activation parameters of the model.

#### Weight Parameters
**<center>Weights without fliers</center>**  

![weight](https://hackmd.io/_uploads/SJwGzRYXA.png)
  
  
**<center>Weights with fliers</center>**  

![Weights with fliers](https://hackmd.io/_uploads/BJZ7zCF70.png)
  


**<center>Histogram of Weights</center>**  

![Histogram of Weights (bitwidth=32 bits)](https://hackmd.io/_uploads/Byq7fCKXR.png)
  




#### Activation Parameters 
**<center>Activation without fliers</center>**  

![Activation](https://hackmd.io/_uploads/HkoEM0tQC.png)
  


**<center>Activation with fliers</center>**  

![Activation with fliers](https://hackmd.io/_uploads/H1WSzAYQR.png)
  


**<center>Histogram of Activation</center>**  

![Histogram of Activation (bitwidth={32 bits)](https://hackmd.io/_uploads/r1etBMRFmC.png)
  

**<center>Histogram of Activation without fliers</center>**  

![Histogram of Clipped Activation (bitwidth={32 bits)](https://hackmd.io/_uploads/H1ZUG0Fm0.png)
  

**<center>Box Plot of Clipped Activation</center>**  

![clipped_acts](https://hackmd.io/_uploads/BJ8DM0KXC.png)
  


### 2. Identify the specific layers or components that quantization impacts the most.
1. 每一個block output activation 的 `mlp.fc1` 數值都會整體偏低，`blocks.9` 的數值偏移最嚴重，中位數為 $-10.152$。
2. Outliers (fliers) 的分布很廣，若是將 outliers 納入計算 quantization 的 S&Z，會造成 large quantization noise。可以使用 clipping(clamping) 來避免這個問題。

### 3. Explain why DeiT-S is harder to quantize than MobileNet. Use relevant charts or graphs to support your findings.

The difficulty in quantizing a network like DeiT-S compared to MobileNet can be attributed to several factors:
1. **Network Complexity**: DeiT-S (Data-efficient Image Transformer - Small) is a vision transformer that uses a transformer architecture, which relies heavily on self-attention mechanisms. The complex attention mechanisms in transformers may not quantize well compared to the simpler convolutional layers in MobileNet.
2. **Sparse Attention Patterns**: Transformers like DeiT-S often exhibit sparse attention patterns, where each token only attends to a few other tokens. This sparsity can make quantization more challenging, as it may result in increased quantization error in the attention weights.
3. **Number of Parameters**: DeiT-S may have a larger number of parameters compared to MobileNet, leading to a higher quantization overhead. Larger models with more parameters are generally more sensitive to quantization.
4. **Gradient Vanishing/Exploding**: Transformers can suffer from gradient vanishing or exploding issues during training, especially when using lower precision weights and activations through quantization. This can affect the convergence and stability of the quantized model.


### 4. Suggestions for improving quantization on DeiT-S. 
1. Apply "Dynamic Range for Activation Quantization" by clipping the activation each layer with the upper limits and lower limits.
    ```=python
    q3 = np.quantile(activation, q=0.75)
    q1 = np.quantile(activation, q=0.25)
    IQR = q3 - q1

    lower_limit = q1 - (IQR * 1.5)
    upper_limit = q3 + (IQR * 1.5)

    np.clip(activation, min=lower_limit, max=upper_limit)
    ```
    The results of the box-plot graph is the same as "Activation without fliers" graph above.
    ![clipped_acts](https://hackmd.io/_uploads/rkYYzCKQC.png)
  
    
    However, due to the clipped value, the distribution of histograms of each layer activation parameter values becomes almost equal, except for some layers which included more outliers, such as the first and second layers. But it should not affect our experiment.
    ![Histogram of Clipped Activation (bitwidth={32 bits)](https://hackmd.io/_uploads/rJbcGRK70.png)



2. Ignore Quantizing Head
    Using PartialXNNPACKQuantizer to ignore quantizing `head` ~~resulting in an improvment of 6 in score. If adding `head_drop` module into ignore list as well, the score rises 10.~~ (update: no improvment for testing whole testloader)

### 5. Explain what you have done in your quantization pipeline to enhance model performance in part 3.

