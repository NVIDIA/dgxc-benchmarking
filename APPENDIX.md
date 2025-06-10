# Appendix

## Glossary of Terms

### 1. **LLM (Large Language Model)**
A type of artificial intelligence model designed to understand and generate human-like text by leveraging vast amounts of data and computational power.

### 2. **Benchmarking**
The process of evaluating the performance of a system, model, or algorithm by comparing it against a set of predefined standards or metrics.

### 3. **Tokenization**
The process of breaking down text into smaller units, such as words or subwords, which are used as input for language models.

### 4. **Inference**
The process of using a trained model to make predictions or generate outputs based on new input data.

### 5. **Fine-tuning**
The process of adapting a pre-trained model to a specific task or dataset by training it further on task-specific data.

### 6. **Hyperparameters**
Configurable parameters that are set before training a model, such as learning rate, batch size, or number of layers.

### 7. **Gradient Descent**
An optimization algorithm used to minimize the loss function by iteratively updating model parameters.

### 8. **Gradient Accumulation (GA)**
A training technique where gradients are accumulated over multiple microbatches before performing a single optimizer step

### 9. **Global Batch Size (GBS)**
The total number of samples processed across all GPUs before a parameter update in a distributed training setup.

### 10. **Microbatch Size (MBS)**
The number of samples processed in a single step(forward+backward only) on a single GPU.

### 11. **DataParallel (DP)**
A parallelization technique where each DP worker works on different input data and the model parameters are updated with an average of the gradients across all the DP workers. For MOE models, DP here referrers to DP on the Attention block.

### 12. **TensorParallel (TP)**
A model parallelization technique that distributes the parameter tensor of an individual layer across GPUs.

### 13. **PipelineParallel (PP)**
A parallelization technique that divides consecutive layers of the model across multiple GPUs, with each GPU processing different layers or segments of the network sequentially.

### 14. **Virtual Pipeline Parallelism (VP)**
A technique that is used to reduce the pipeline bubbles or a measure of the time the GPUS spend idle when pipeline parallelism is enabled. See more information [here](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/parallelisms.html#interleaved-pipeline-parallel-schedule)

### 15. **Fully Sharded Data Parallel (FSDP)**
A parallelism technique that enables GPU memory savings by sharding the model parameters, gradients, and optimizer states across GPUs

### 16. **Model FLOP Utilization (MFU)**
A metric that measures the efficiency of GPU usage in AI computations by comparing the observed throughput to the theoretical maximum performance of the hardware.

### 17. **Expert Parallelism (EP)**
The model parallelism strategy used in Mixture of Experts (MoE) architectures, that distributes experts of an MoE model across GPUs.

### 18. **Expert Tensor Paralellism (ETP)**
The model parallelism strategy for doing Tensor Parallelism for MoE layers.  
