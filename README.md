# Optimization for LLM Inference

### Goals
This project aims to compare the multiple strategies that optimizes the llm inference. The optimization techniques include 1) Naive LLM, 2) LLM with quantization (half precision), 3) Flash Attention2 and Specuative Streaming

### Components

1) datasets directory include all codes related to the dataset.

2) models include all codes for optimization techniques that mentioned above

3) As speculative streaming needs the modification in both LLM code and generation, we modified the code 
from huggingface and huggingface directory includes all modification and codes for speculative streming.

Specifically, the model code is in huggingface/transformers/models/gpt_neo and generation code is included in huggingface/transformers/generation/utils.py.

### Executing program

Please use the following command

```
python dataset_tps.py --model_type [optimization technique]
```

### Results

When running with GeForce RTX 3090, the results are as below. (tokens/sec)

|             | Spider     |       DialogSum         |   E2E-NLG|
|-------------|-----------------|----------------|-----------------|
|        Naive LLM    | 44.459 | 45.504 | 45.560 |
| LLM+half precision Quantization       | 41.660      | 42.075     | 42.360 |
| LLM+Flash Attention2 | 49.229          | 49.290          | 43.971     | 
| LLM+Speculative Streaming | **54.661**          | **56.038**         | **54.782**    | 

When running with A100, the results are as below. (tokens/sec)

|             | Spider     |       DialogSum         |   E2E-NLG|
|-------------|-----------------|----------------|-----------------|
|        Naive LLM    | 46.530 | 47.907 | 48.372 |
| LLM+half precision Quantization       | 44.852     | 44.879    | 44.421 |
| LLM+Flash Attention2 | 57.167         | 56.863         | 56.748     | 
| LLM+Speculative Streaming | **58.114**          | **58.937**         | **58.150**    | 


The performance by acceptance ratio was as a below graph.

![image](images/HPC6998.png)

The time comsumption for each step was as a below graph.

![image](images/HPC_STACKED.png)

We can notice that LLM+Speculative Streaming was the most powerful optimization technique and unsuprisingly the performance gain is larger when the token accept ratio is higher.

The result also demonstrates 1) Speculative streaming spends comparingly large preparation time compared to other techniques due to other stuffs such as building tree structure, verifying the speculated tokens, 2) even with large preparation time, speculative streaming spends the least total time thanks to less forward time which indicates that the benefit from running fewer forward steps (as one forward step generates more tokens) outweighs the more computation for each forward step.

You can find more details / experiment on other GPUs on our final report.

## Notice
This code is compatible with python3.9

## Version History

* 0.1
    * Initial Release (6 May 2024)