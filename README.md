## Multimodal LLMs with Database Constrained Decoding for Recycling Classification
**Andrew Luo**

This repo includes the code for the project Multimodal LLMs with Database Constrained Decoding for Recycling Classification.

We investigate the usefulness of multimodal LLMs for the task of large-scale classification. We evaluate on visual recycling classification, where given an image, the task is to determine the best match in a city database with hundreds of classes. Since it is not immediately obvious how to adapt a multimodal LLM for this task, as all the classes may not fit into the model's context length, we propose a new method called **database constrained decoding**. DCD is an inference-time technique which limits each step of decoding such that the current output is guaranteed to lead to some element in a database.

We also include the **Waste Wizard Dataset** at [this HuggingFace url](https://huggingface.co/datasets/acluous/waste-wizard-materials-list). The dataset contains city recycling databases spanning three different countries as well as an evaluation set of images labeled with the corresponding database item for each city.  In the below table, we show the performance of different vision-language models on this Waste Wizard Dataset, where we find that a multimodal LLM combined with DCD outperforms a contrastive model on large real-world databases. We also include the result for a small toy database containing ten classes.

| Method                                                 | Toy-10  | Davis-361 | Mountain View-470 | Waverley-349 | Waterloo-1094 |
|--------------------------------------------------------|---------|-----------|-------------------|--------------|---------------|
| Idefics2 - DCD (Ours)                                  | 0.70    | **0.58**  | **0.62**          | **0.58**     | 0.38          |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | **0.93**| 0.57      | 0.53              | 0.56         | 0.38          |  

## Demo
To try our method, press the "Open In Colab" button below.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acluous/recycling-database-constrained-decoding/blob/main/demo.ipynb)

