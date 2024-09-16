# A Lightweight ControlNet for Fine-Tuning Stable Diffusion on Colab GPU
 
## 1. Motivation

Deep learning in computer vision has demonstrated remarkable achievements. Diffusion models, in particular, have shown outstanding performance in generative AI. However, these models often come with significant drawbacks—they require large-scale datasets and high-performance hardware, leading to substantial resource consumption.

For instance, deep learning models store numerous parameter values, resulting in large file sizes. In practical scenarios, loading a deep learning model demands significant GPU memory. Limited memory makes it difficult to load the model directly, and leveraging the stored parameters for numerous computations can be time-consuming, especially with extended processing times for image processing and inference.

We propose optimizing and making ControlNet lightweight to enhance resource efficiency. Our goal is to modify the model to operate effectively, even in resource-constrained environments.

### Contribution

- **Model Size Reduction**: We modified the basic structure of ControlNet to design a lightweight model with fewer parameters, reducing memory requirements and making deployment more efficient.
  
- **Inference Speed Improvement**: By lightweighting ControlNet, we improved the model's inference speed, crucial for real-time applications.

- **Resource Efficiency**: We verified whether the lightweight ControlNet could operate effectively in memory and computation-constrained environments, ensuring its practicality.

- **Performance Analysis**: We compared the performance of the lightweight ControlNet with the original ControlNet to assess the impact of the modifications, ensuring performance was maintained or enhanced while maximizing resource efficiency.

---

## 2. Background
<img width="680" alt="Screenshot 2024-09-16 at 7 10 10 PM" src="https://github.com/user-attachments/assets/b320dcd0-d245-4a6f-803e-70878f65b369">

ControlNet is a Text-to-Image Diffusion model that generates images from text and additional conditions, such as sketches, depth maps, edge maps, human poses, and more.

It maintains the generative performance of a pretrained large-scale Diffusion model and achieves good learning performance even with limited training data. The model is effective when generating images under new conditions without requiring full retraining. With open-source code and datasets, experimentation becomes convenient, allowing for multiple trials.
<img width="845" alt="Screenshot 2024-09-16 at 7 10 28 PM" src="https://github.com/user-attachments/assets/3bb49a8e-6fbe-4a4c-8620-29753594aba3">

To generate images under new conditions, ControlNet copies the parameters of a pretrained diffusion model, forming parallel copies referred to as "Locked Copy" and "Trainable Copy." These two copies are connected by a zero convolution layer.

- **Locked Copy**: Represented by the left gray box, this part preserves the existing parameter weights of the pretrained model. As the name suggests, the parameters are fixed and do not change during training.

- **Trainable Copy**: Represented by the right gray box, this part is used to learn additional conditions from the training data. The parameters are updated during training to adapt to the specific conditions.

---

## 3. Method

### How We Do It
<img width="812" alt="Screenshot 2024-09-16 at 7 10 46 PM" src="https://github.com/user-attachments/assets/5bacab74-88dc-472c-b653-3cf42fff3702">

The original ControlNet model uses the encoder portion of the U-Net architecture for training. In our modification, we incorporate pixel-wise MLPs (Multilayer Perceptrons) in this section. These MLPs pass through linear layers, utilize average pooling, and inject the result into the Stable Diffusion (SD) Encoder Block using zero convolution.

While using MLPs reduces the number of parameters, potentially affecting performance, it addresses memory limitations and improves training speed.

---

## 4. Implementation

### 4.1 Data

We trained ControlNet on the **Fill50k dataset**, which consists of source images, target images, and prompts. ControlNet generates images based on a given prompt, with the goal of producing images that match the shape of the source image.

- **Prompt**: Natural language sentences describing the task.
- **Control**: Image condition data tailored for the task (e.g., the Canny Edge task).

### 4.2 Improvements on System Resources
<img width="454" alt="Screenshot 2024-09-16 at 7 28 41 PM" src="https://github.com/user-attachments/assets/3d27f01b-c131-4a5a-821e-675d6e8976d3">
<img width="532" alt="Screenshot 2024-09-16 at 7 23 57 PM" src="https://github.com/user-attachments/assets/88769a8a-0a8b-42f2-aa5d-dee081e34547">

The training results are displayed in the following order: Prompt, Control, Reconstruction, and Samples. Training has been conducted for up to 2 epochs, with each epoch consisting of 12,500 time steps.

- **Reconstruction**: Images generated by a pretrained stable diffusion model based solely on text.
- **Samples**: Images generated by the MLP that we trained, corresponding to the trainable copy in the ControlNet structure.

### 4.3 Improvements in Performance

- **Modified ControlNet**:  
  <img width="541" alt="Screenshot 2024-09-16 at 7 24 16 PM" src="https://github.com/user-attachments/assets/e0a1ec3d-d34f-44b8-87c0-38190ceb9ca9">
Before training, the images generated were entirely unrelated to the control, although some color reflection from the prompt could be observed. The images appeared to be generated using only the pretrained stable diffusion model based on the prompt, as the control had not been trained.
<img width="540" alt="Screenshot 2024-09-16 at 7 24 35 PM" src="https://github.com/user-attachments/assets/77fe48aa-574b-4b17-8ec0-7c732b89660d">

  After training for one epoch, the generated images showed edges more similar to the control, but the colors were still not fully aligned.
<img width="597" alt="Screenshot 2024-09-16 at 7 24 47 PM" src="https://github.com/user-attachments/assets/a5870a63-4e10-41a3-bbe3-df0676343fd3">

  After 2 epochs, the results became more similar to the control, but there were still cases where the generated images differed. Due to constraints in the Colab environment, further training for increased accuracy was not possible, but rough comparisons with the original ControlNet were made.

- **Original ControlNet**:  
  <img width="736" alt="Screenshot 2024-09-16 at 7 25 00 PM" src="https://github.com/user-attachments/assets/bf9d0de0-91aa-492d-b6de-fa02ac65647c">
Due to resource constraints, we referred to the results of another experiment instead of running our own. Despite the limited training, the original ControlNet showed much more accurate performance compared to the modified version. We concluded that achieving performance comparable to the original model when lightweighting would require extended training.

---

## 5. Conclusion

We built a scalable model that can be fine-tuned on Colab GPU. Instead of using the exact same architecture as Stable Diffusion for the "trainable copy," we built an MLP for a lightweight model. This allowed us to fine-tune Stable Diffusion to learn task-specific input conditions.

### Limitations

While we achieved clear advantages in terms of memory efficiency, a drawback was the relatively longer training time required to match the accuracy of the original model. There is a trade-off between GPU RAM size and training time, as well as a potential decrease in expressive power for more complex datasets.

### Discussion

Further research is required to understand how model performance degrades with increasing dataset complexity. ControlNet should reflect both the prompt and the control effectively. Without quantifiable evaluation scores, and because the paper used surveys for accuracy comparison, we regret not conducting a similar survey ourselves.

## Reference
We refered to the code from here.

Reference experiments and findings are described here: https://medium.com/@dzvinkayarish/controlnetlite-smaller-and-faster-controlnet-55b26cef946e .
This project was inspired by this discussion https://github.com/lllyasviel/ControlNet/discussions/188 .

