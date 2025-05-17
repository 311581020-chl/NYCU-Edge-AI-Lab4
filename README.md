# **Lab 4: Quantize DeiT-S**

<span style="color:Red;">**Due Date: 5/22 23:55**</span>

## Introduction

This lab aims to quantize the DeiT-S model, resulting in reduced model size and potentially improved inference speed without significant loss in accuracy.

## Part 1: Simple Quantization Pipeline (0%)

* TAs will provide the naive pipeline of quantizing DeiT-S. Use part 1 as your reference for building your own quantization pipeline.

    > [lab4_v2.ipynb](https://colab.research.google.com/drive/165iRcs21tKN3MvxBXpPZLfttZntYM8Og?usp=drive_link)
    
* Below is a DeiT-S model with 90.99% accuracy on **CIFAR100**, finetuned by TAs. You will be using this model as a starting point for quantization:

    > [0.9099_deit3_small_patch16_224.pth](https://drive.google.com/file/d/1zhUJyCPkSSguggUtYOdv9OQBAy4fEf4h/view?usp=drive_link)

    You can load the above model with the following snippet:
    ```python
    import torch

    model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location="cpu")
    ```


## Part 2: Analysis (60%)

You might observe a significant accuracy drop after running through the simple quantization pipeline in Part 1. Recalling what you've learned in class, identify the cause and develop an optimal quantization method for DeiT.

In this lab, you will write a brief report in **HackMD**, explaining the reasons for the substantial accuracy drop after quantization. The report should include the following:


1. A visual analysis of the weight/activation parameters of the model. **(20%)**
2. Identify the specific layers or components that quantization impacts the most. **(10%)**
3. Explain why DeiT-S is harder to quantize than MobileNet. Use relevant charts or graphs to support your findings. **(10%)**
5. Suggestions for improving quantization on DeiT-S. **(5%)**
6. Explain what you have done in your quantization pipeline to enhance model performance in part 3. **(15%)** *(Top methodologies will be showcased by TAs in DC as a learning resource for the class)* 


## Part 3: Quantizing DeiT-S (40% + Bonus 5%)

Based on your analysis of DeiT-S, refine the quantization pipeline from Part 1 to improve its performance.

1. Quantize DeiT-S using your enhanced pipeline.
2. Submit the final quantized model and your code to e3.

**(Below is optional, Bonus 5%):** 
1. Convert the quantized DeiT-S model to Executorch (.pte).
2. Deploy the quantized model on a Raspberry Pi.
3. Contact the TA on Discord to schedule a demo if you complete this.

> **Note**
> [We can’t do performance evaluation now since the model is not lowered to target device (Executorch, *.pte*), it’s just a representation of quantized computation in ATen operators.](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html#checking-model-size-and-accuracy-evaluation)

## Hand In Policy

You will need to hand-in:
* your quantized model ***deits_quantized.pth***
* your quantization pipeline that generates your final quantized model
* **url.txt** should include the URL of your HackMD report.

Please organize your submission files into a zip archive structured as follows:

```scss
YourID.zip
    ├── model/
    │     └── deits_quantized.pth
    │ 
    ├── YourID.ipynb (or YourID.py)
    └── url.txt
```

## Evaluation Criteria

1. TAs will use the following code **written in *lab4_v2.ipynb*** to load your quantized model and calculate the final score:
    ```python
    score = lab4_cifar100_evaluation(quantized_model_path='model/deits_quantized.pth')
    ```
2. Your score is determined by the **size** and **accuracy** of the quantized model. The evaluation function is calculated as follows:
    ```python
    score = 0
    if model_size <= 30: score += 10
    if model_size <= 27: score += 2 * math.floor(27-model_size)
    if acc >= 86:
      score += 10 + 2 * math.floor(acc-86)
    ```