
# Literature Review

## Introduction
Automating attendance tracking through face recognition technology has emerged as a powerful solution to eliminate manual entry errors. Traditional systems often rely on physical interactions, which can be time-consuming. The rise of **Jetson Nano** makes it feasible to deploy **real-time face recognition** systems on low-power devices. This literature review discusses key techniques, algorithms like **FaceNet** and **BlazeFace**, and evaluates their use in developing an efficient face recognition-based attendance system.

## State-of-the-Art Techniques in Face Detection and Recognition

### FaceNet: A Deep Learning Approach to Face Recognition (2015)
FaceNet, developed by Schroff et al., utilizes a deep learning-based approach for face recognition. The model employs a triplet loss function to map face images into a Euclidean space where distances indicate the similarity between faces. It has shown impressive results, achieving **99.55%** accuracy on the **LFW (Labeled Faces in the Wild)** dataset, making it a highly accurate method for face recognition.

### BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs (2020)
BlazeFace, developed by Google Research, offers efficient face detection at high speeds, making it suitable for mobile and embedded platforms like **Jetson Nano**. It can handle up to **1000 frames per second**, making it ideal for real-time video feeds in crowded environments. Its low power consumption ensures it’s a great fit for embedded systems requiring rapid face detection.

### A Survey on Face Recognition Techniques: State-of-the-Art Methods and Applications (2021)
This survey explores various face recognition techniques, including **FaceNet**, **VGGFace**, and **ResNet**. The paper emphasizes the need for **efficient algorithms** suitable for **low-power devices**, such as **Jetson Nano**. It also discusses optimization methods like **TensorRT** that reduce inference times, essential for real-time applications.

### Real-Time Face Recognition on the Jetson Nano for Smart Attendance (2023)
This paper directly addresses the deployment of **FaceNet** on **Jetson Nano** for real-time attendance tracking. Using **TensorRT optimization**, the system achieves **20 faces per second**, making it suitable for environments with up to **50 individuals**.

### Local Binary Patterns for Face Recognition (2015)
Before the advent of deep learning, **LBPH (Local Binary Pattern Histogram)** was a commonly used technique for face recognition. While not as accurate as newer methods like **FaceNet**, LBPH remains a viable option in constrained environments with fewer people or controlled settings, thanks to its simplicity and low computational requirements.

## Feasibility for Semester Project
The goal of this project is to develop a real-time face recognition-based attendance system using the **Jetson Nano**. The algorithms that will be employed are:

- **FaceNet**: Chosen for its high accuracy and ability to generate discriminative facial embeddings.
- **BlazeFace**: Selected for fast face detection, crucial for real-time applications.
- **LBPH**: Suitable for low-complexity, resource-constrained environments where real-time processing is not critical.

These techniques are compatible with **Jetson Nano**, which has sufficient power to run these models while maintaining low energy consumption, making it ideal for edge-based face recognition systems.

## Research Gaps and Contribution
Most research has focused on large-scale systems or cloud-based solutions. This project aims to address the gap in deploying face recognition systems on **edge devices** like the **Jetson Nano**, optimizing algorithms for small-to-medium environments, and providing insights into using embedded systems for real-time face recognition applications.

## References
- **S. Bussa et al.**, "Smart Attendance System using OPENCV based on Facial Recognition," 2020.
- **E. Yose et al.**, "Portable smart attendance system on Jetson Nano," Bulletin of Electrical Engineering and Informatics, vol. 13, no. 1, 2024.
- **M. Jagli et al.**, "Advancements in Facial Recognition for Automated Attendance Systems," Preprints, 2024.
- **V. Bazarevsky et al.**, "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs," Google Research, 2020.
- **S. Chen et al.**, "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices," 2020.



