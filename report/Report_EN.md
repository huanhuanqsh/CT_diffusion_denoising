# Report on Training of CT Image Denoising Diffusion Models Based on Open-Source Project Dn-Dp

- [Introduction](#Introduction)
- [Materials and Methods](#MATERIALS-AND-METHODS)
- [Results](#Results)
- [Discussion](#Discussion)
- [Conclusion](#Conclusion)



## Introduction

In medical imaging, Low-Dose Computed Tomography (CT) is favored due to its low radiation exposure, yet introduces increased noise, compromising diagnostic accuracy and reliability. Denoising low-dose CT images is therefore crucial. Supervised deep learning methods have advanced in this field, presenting solutions. This report leverages the open-source project "Dn-Dp" [[1]]([1]) to introduce a novel zero-shot denoising approach for low-dose CT images via diffusion probabilistic priors, validated through experimental training.

## Materials and Methods

### Model Overview

Conventional supervised deep learning methods require large datasets of low-dose and full-dose CT pairs, challenging to obtain in clinical settings. Unsupervised methods often demand extensive low-dose data or tailored acquisition procedures. Addressing these constraints, a novel unsupervised method is proposed, leveraging only full-dose CTs during training for denoising low-dose images.

The approach initiates with a cascade of unconditional diffusion models that generate high-quality, high-resolution CTs from low-resolution inputs, enhancing the feasibility of high-resolution model training. Low-dose CTs are then introduced into the reverse diffusion process with a MAP framework informed by the diffusion priors for iterative denoising, adapting to varying noise levels with adaptive λ balancing strategies.

### Training Procedure

- **Environment Setup**: 

  - Ubuntu 20.04, Intel(R) Xeon(R) Platinum 8260C CPU @ 2.30GHz
  - NVIDIA RTX 3090 with 24GB VRAM
  - Python 3.10, Pytorch 2.1, CUDA 11.8, dependencies installed via `pip install -r requirements.txt`.

- **Data Preparation**: 

  The dataset utilized in this study comprises two components:

  - First is a set of paired low-dose CT images and their corresponding high-dose CT images, which serve as the basis for training and validating the denoising capability of the diffusion model.

  - Second is an independent test set consisting of low-dose CT images alone, employed for the final assessment of the model's performance. All images are sourced from clinical data, encompassing abdominal CT scans, thereby ensuring the model's generalization performance. Preprocessing steps, including standardization, normalization, and resizing, have been applied to all images to conform to uniform input requirements.

- **Parameters**: 

  - Adjust training parameters such as model paths, data preprocessing dimensions, DDIM configurations, and denoising parameters using YAML or JSON configuration files.
  - Test images are placed in a designated `input_folder`, with outputs saved in the `output_folder`.

- **Training**: The super-resolution and unconditional diffusion models are trained utilizing the scripts `sr_training.py` and `sample_training.py`, respectively. To enhance the training efficacy of high-resolution models, the following strategies are adopted:

  - **Patching Technique Introduction** [[2]], akin to the patching transformation seen in ViT architectures, is employed to reduce sampling time and memory consumption. This approach breaks down images into smaller patches for more efficient processing.
  - **Progressive Distillation for Sampling Optimization** involves gradually reducing the number of sampling steps per iteration by halving them until the model can maintain comparable denoising quality with fewer steps. This strategy accelerates training while preserving model performance.
  - **Enhanced Deep U-Net Architecture** for high-resolution models, where the depth of the U-Net structure is increased, facilitating better feature extraction and representation learning at higher resolutions, thereby improving overall model capacity and performance.

### Optimization

- **Testing Preprocessing**: For non-png or non-unit16 images, custom read/write functions are adjusted.

  In the context of this report, where the test data consists of uint8 images, the following adaptations have been implemented:

  ```python
  def read_a_img(path):
      return imageio.imread(path) / 255.
  
  def save_a_img(img, path): 
      return imageio.imwrite(path, (img*255).astype(np.uint8))
  ```

- **Metrics**: FID, PSNRMS-SSIM, PSNR evaluate denoising.

- **Hyper parameter Tuning**: Monitoring metrics, adjusting λ0, a, b, c for optimal denoising.

### Testing & Evaluation

- **Training Progress Metrics**: 

  - Unconditional diffusion: l_pix=1.3170e-04, models: *128_I200000_E5000_gen.pth、128_I200000_E5000_opt.pth*

  - Super-resolution: PSNR=5.3642e+01, models: *128_512_I100000_E625_gen.pth、128_512_I100000_E625_opt.pth*

- **Denoised Images**: Code provided for testing:

1. Apply the unconditional diffusion training model by executing `denoising.py -c ./config/Dn_128.yaml`, to perform noise reduction on specified low-resolution CT images. This process generates 128*128 low-resolution images, which are then saved in a designated output directory. Optionally, metrics can be computed to assess the model's performance, as illustrated in the following figure:

   ![L067_0001](./images/test.png)

2. `denoising.py -c ./config/Dn_128_512.yaml` for high-resolution denoising conditioned on low-res denoised images.

## Results

The project successfully demonstrates diffusion models' application in low-dose CT denoising, including:

- **Denoising Quality**: Clear noise reduction in low-resolution abdominal CTs retaining diagnostic detail.
- **High-Resolution Handling**: Effective processing of high-resolution CTs conditioned on low-res denoised images, highlighting cascade potential.
- **Efficiency**: Patching significantly reduces sampling time for practical deployment in resource-limited environments.
- **Model Outcomes**: Results from [this report's trained models](https://github.com/huanhuanqsh/CT_dd_model) are available for further research and application.

## Discussion

This study showcases the potential of diffusion models for CT denoising, especially in low-dose scenarios, overcoming traditional methods' reliance on paired datasets. Patching enhances practicality, reducing computational costs, yet high-performance hardware remains necessary.

**Comparison & Insights**: This study addresses the direct impact of reducing iterations on efficiency in diffusion models and provides solutions beyond per-step cost optimization. It offers a fresh perspective on efficiency improvements and highlights the importance of model flexibility in parameterization and design for future researchers.

**Report Limitations**: Pending issues with the high-resolution model's validation are under investigation.

**Future Directions** include exploring model optimizations, flexible patching strategies, broader application across modalities, and further studies on generalizability, including cross-device and dose-level consistency.

## Conclusion

The probabilistic prior-based diffusion models effectively denoise low-dose CT images, requiring no training data, showcasing their vast potential in medical imaging. By incorporating patching, the project ensures enhanced performance and efficiency in real-world deployment, marking significant advancements for low-dose CT in clinical practice.

**References**

[1] DeepXuan/Dn-Dp GitHub Repository (2023). Available at: https://github.com/DeepXuan/Dn-Dp

[2] Improving Diffusion Model Efficiency Through Patching. Available at: https://arxiv.org/abs/2207.04316

[3] Progressive Distillation for Fast Sampling of Diffusion Models. Available at: https://arxiv.org/abs/2202.00512

[4] Pre-trained Cascaded Diffusion Models Download. Available at: https://drive.google.com/drive/folders/1sHWtDlUCO-4cb-v_ijR1i3c2PYqB44xs?usp=sharing
