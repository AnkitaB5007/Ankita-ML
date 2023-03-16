---

date: 2023-03-15
---

# title: "How to learn a single prompt that can be used for multiple tasks efficiently"
### Introduction

Finetuning pretrained language models (PLMs) has led to significant improvements across various downstream NLP tasks (Devlin et al., 2019; Howard & Ruder, 2018; Raffel et al., 2020).

### Problems:
+ Conventional paradigm of full task-specific fine-tuning(FT); difficult to scale to multiple task
+ PLMs have billions of parameters 
Recent interest emerges in parameter-efficient methods for model tuning  (Houlsby et al., 2019; Lester et al., 2021; Ding et al., 2022), where the goal is to learn only a small number of additional parameters per task while achieving performance comparable to full finetuning.

### Approaches:
+ Prompt tuning (PT), which prepends tunable continuous prompt vectors to the input, has emerged as a promising approach for parameter-efficient transfer learning with PLMs (Liu et al., 2021a; Li & Liang, 2021; Lester et al., 2021; Liu et al., 2022b; 2021b). 
+ PT freezes the PLM parameters and only learns a small set of task-specific prompt vectors.
+ However, despite their impressive performance, there is still a large gap between prompt tuning and full finetuning (Lester et al., 2021). Additionally, this approach is sensitive to initialization and often requires more training time than finetuning (Su et al., 2022; Zhong et al., 2022).

