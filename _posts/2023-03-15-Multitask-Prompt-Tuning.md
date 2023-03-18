---

date: 2023-03-15
---

# How to learn a single prompt that can be used for multiple tasks efficiently
### Introduction

Finetuning pretrained language models (PLMs) has led to significant improvements across various downstream NLP tasks (Devlin et al., 2019; Howard & Ruder, 2018; Raffel et al., 2020).

### Problems:
+ Conventional paradigm of full task-specific fine-tuning(FT); difficult to scale to multiple task
+ PLMs have billions of parameters 
Recent interest emerges in parameter-efficient methods for model tuning  (Houlsby et al., 2019; Lester et al., 2021; Ding et al., 2022), where the goal is to learn only a small number of additional parameters per task while achieving performance comparable to full finetuning.

### Visualizing the Problem
 
We have a set of source tasks  $\bm{\textit{S}} = {S_{1}, S_{2}, ... , S_{k}}$ and a set of target tasks  $\textit{T} = {T_{1}, T_{2}, ... , T_{k}}$.

We want to learn a single shared prompt matrix that can perform well on target tasks. The general approach would be to learn a single shared prompt and finetune the prompt on each task in the target task set. This does not align with the initial idea of having a single prompt that can 
	+ neither adapt to multiple target tasks in a parameter-efficient way 
	+ nor learns anything about the similarities between source tasks. 


### Approaches:
+ Prompt tuning (PT), which prepends tunable continuous prompt vectors to the input, has emerged as a promising approach for parameter-efficient transfer learning with PLMs (Liu et al., 2021a; Li & Liang, 2021; Lester et al., 2021; Liu et al., 2022b; 2021b). 
+ PT freezes the PLM parameters and only learns a small set of task-specific prompt vectors.
+ However, despite their impressive performance, there is still a large gap between prompt tuning and full finetuning (Lester et al., 2021). Additionally, this approach is sensitive to initialization and often requires more training time than finetuning (Su et al., 2022; Zhong et al., 2022).

### Current Approach- Dubbed MPT
In the previous approaches, soft prompts are individually learned per task using various approaches, particularly vanilla prompt tuning (see Vanilla prompt tuning). This phase of the process is called source training. In source training, we aggregate the prompts we learned from the set of source tasks. In the next phase, target adaptation, we adapt a prompt from the pool of aggregated pre-trained (on source tasks) prompts and initialize the prompt for further fine-tuning on a target task based on a (potentially learned) similarity measure.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226112617-20eb9806-b5b2-4555-9c1a-12829eba696d.png" alt="***Figure 2***.An illustration on prompt decomposition for two tasks."/>
</p>

In the dubbed version of Multitask prompt tuning (MPT), we follow the same strategy of Source training and target adaptation with a slight twist. It seems that simply sharing the aggregated soft prompts in the Vanilla version did not fit the target task so well and resulted in a gap in the performance of the PLM w.r.t Source and Target tasks. 
#### Source Training:
The primary goal of the authors was to design a single shared prompt matrix $**\widehat{P^*}**$ that could easily adapt to the nature of the target tasks. They try to model this single prompt matrix as a combination of two superpowers. First one, it should carry all the common flavors from the source tasks (RTE, QA, Text summarization, etc.), and second, it should be good at a specific task(say, for a QA task). How do they learn this single shared prompt matrix, is the "**big question**"?
Well, they literally "*learn*" this prompt from a very popular and common concept in Machine Learning which is *Knowledge Distillation*. If we are talking about KD, we must have Teachers and Students! 

We already have a bunch of soft prompts from the source tasks, and these soft prompts per task will be the teachers. Each Teacher soft prompt is an ‘expert’ on the task it was pre-trained upon. This knowledge has to be distilled, compressed, and delivered to the student prompt. For a task *k* in the source task set, we have a teacher prompt $ P_k $ and a student prompt $**\widehat{P_k}**$. The student prompt is expected to get the gist from all the teacher prompts (single shared matrix, our primary goal to learn) and also perform well on the task it was designed for initially(task-specific matrix).

In parallel, these soft prompts will be decomposed into two matrices A and B(Prompt decomposition). We decompose the soft prompt P_k for the k-th task into two parts, as shown in Figure 3. Let ***P^∗*** ∈ R l×d denote the shared prompt across all tasks, and further let u_k ∈ R l , vk ∈ R^d be the task-specific vectors for each task k. The task-specific vectors form a rank-one matrix $ W_k = u_k ⊗ v_k^T $ , which has the same dimensions as the shared prompt ***P^∗*** . The task prompt 
$\hat{P} $ 
for k-th source task is then parameterized as:

<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226111397-72205dba-d768-4152-a7d0-d12f84cf31b2.gif" alt="Equation 1"/>
</p>

![Equation 1]()

General information across the set of source tasks ***S*** can be captured by “slow” weights ***P^**** shared across tasks, while the “fast” weights W_*k* could then encode task-specific knowledge for S_*k* in a low-rank subspace.

#### Losses
#### Target Adaptation
