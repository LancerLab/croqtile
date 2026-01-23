# What is Choreo?

## Choreo's Target


Choreo is a DSL designed to **orchestrate DMA data transfers** with ease. It abstracts away the complexity of managing DMA operations by offering a high-level, declarative programming model. Developers can use Choreo to specify when and how data should be transferred between different memory regions, while leaving the low-level details of memory management to Choreo.

Choreo was developed to simplify the orchestration of DMA transfers while maintaining the performance and flexibility needed for modern computing systems. By providing an easy-to-use interface, Choreo allows developers to focus on high-level memory orchestration instead of dealing with the intricacies of hardware-specific programming.

## Introduction to DMA and the Need for Orchestration

Direct Memory Access (DMA) plays a critical role in high-performance computing systems, allowing peripherals, processors, and other hardware components to access memory directly, without the need for the CPU to be involved in data transfer. This capability is especially crucial in systems like GPUs, network cards, and embedded systems, where large volumes of data need to be moved quickly between memory and devices.

However, managing DMA transfers efficiently can be complex. It involves coordinating data movement across different memory spaces, ensuring synchronization, and minimizing latency. Without an effective mechanism to orchestrate DMA operations, systems can quickly become bottlenecked by inefficient data movement.

## Key Features of Choreo

- **High-Level Abstractions**: Choreo simplifies complex DMA orchestration with concise syntax, reducing the boilerplate code required for memory transfers.
- **Interoperability**: Choreo integrates seamlessly with other kernel programming models like **CUDA/Cute**, and others, making it ideal for use in heterogeneous environments.
- **Optimized for Performance**: Choreo abstracts the complexity of DMA, enabling high-performance memory transfers without requiring manual fine-tuning.
- **Ease of Integration**: Choreo can be integrated into existing systems, allowing for simplified memory management in both standalone and hybrid kernel programming environments.
