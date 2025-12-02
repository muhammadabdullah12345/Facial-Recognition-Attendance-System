# Feasibility of Algorithms on Jetson Nano

The **Jetson Nano** is a powerful yet low-energy platform, ideal for deploying **real-time face recognition** systems. Here's why the selected algorithms are feasible for use on this platform:

- **FaceNet**: Although FaceNet is computationally expensive, it can be optimized using **TensorRT** on the **Jetson Nano** to reduce inference times significantly.
- **BlazeFace**: BlazeFace is lightweight and optimized for mobile GPUs, making it suitable for Jetson Nano. Its ability to handle 1000 frames per second allows it to process faces in real-time without overloading the system.
- **LBPH**: For small-scale systems or controlled environments, LBPH offers a **computationally efficient solution** with low memory requirements, making it perfect for Jetson Nano when processing fewer faces.

The **Jetson Nano** provides enough processing power to support these algorithms while maintaining **low power consumption**. This makes it a highly suitable choice for **edge-based face recognition systems**.
