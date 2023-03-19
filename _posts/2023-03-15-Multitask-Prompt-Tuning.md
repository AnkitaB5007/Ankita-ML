---

date: 2023-03-15
---

# How to learn a single prompt that can be used for multiple tasks efficiently
In this blog post, we will go over the ICLR 2023 paper titled MULTITASK PROMPT TUNING ENABLES PARAMETER-EFFICIENT TRANSFER LEARNING. Recently, I was reading few articles about **"Prompt tuning"** where the idea of having a compact model that can fit to any target downstream tasks, seemed quite interesting. This paper cannot stand alone without the contribution of two key works published previously, (Houlsby et al., 2019, Vu et al., 2021).

**Goal of this blog post**: The main goal with this blog post is to provide a foundation for 
1. Parameter efficient Prompt tuning utilizing Transfer Learning capabilities 
2. Soft prompt transfer 
3. How similarity between various tasks can benefit in prompt design 
4. **Multitask Prompt tuning** : Having a nice compact prompt matrix(few tunable parameters to worry about) to adapt to any target task.

Outline of this Blog Post


### Idea Behind Compact Models
We don't want to train the entire model (billions of parameters) for every new task(Model tuning). This brings us to the idea of having compact and extensible downstream models.  Compact models are those that solve many tasks using a small number of additional parameters per task. Extensible models can be trained incrementally to solve new tasks, without forgetting previous ones. This can be acheived via **Transfer learning** strategies like : 
1. Feature-based : Feed the custom downstream models with task pre-trained real valued embedding vectors (word/sentence/paragraph level)
2. Finetuning : Usually, better than the previous one. Simply copy the weights from a pre-trained model and train them on a downstream task. Vanilla fine-tuning does require a new set of network weights for every new task. 
      + Prompt Tuning freezes the PLM parameters and only learns a small  set of task-specific prompt vectors.
     + Very sensitive to initialization
     + Can take very long time


### How do we make Transfer learning parameter-efficient?
In case of Finetuning, one way could be to share parameters in the lower layers and tune the higher layers as per the target task. We do not wish to touch the model parameters(i.e. freeze them) and only try to optimise the task-specific parameters(Prompt tuning) so that we can be parameter-efficient. Adapter tuning strategy(Houlsby et al., 2019) tries to do the same by injecting new layers into the original network. The weights of the original network are untouched, whilst the new adapter layers are initialized at random. Another benefit of using adapter modules is that you do not need simultaneous access to the source tasks(in case of Multitask prompt tuning). 

Another way is to learn optimisable prompts, also called soft because they are indeed '*soft*' as they come from the model's vocabulary but over the process they change their form by gradient descent. These soft prompts can be transferred to fit any kind of target task.

### Soft Prompts: How do we learn them and transfer them?
I throughly enjoyed reading the paper **SPoT**(Vu et al., 2021) which talks about soft prompt transfer. They basically, learn a prompt on one or more source tasks and then use it to initialize the prompt for a target task(remember, base model is frozen). The following figure gives an excellent overview of their approach:
<p align= 'center'>
    <img src="https://user-images.githubusercontent.com/39300414/226206791-83283787-10b8-4ad6-a04a-6692c821159c.JPG" alt>
    <em>Figure 1: An illustration of our task-agnostic (left) and task-specific (right) SPOT approaches. Left: We learn a
	single prompt on one or more source tasks, which is then used to initialize the prompt for each target task. Right:
	We learn prompts for source tasks, and save early checkpoints as task embeddings and best checkpoints as source
	prompts. These form the keys and values of our prompt library. Given a novel target task, a user: (i) computes a
	task embedding, (ii) retrieves an optimal source prompt, and (iii) trains a target prompt, which is initialized with
	the source prompt.</em>
</p>
They treated task prompts as "*task embeddings*" to construct a semantic space of tasks and formalize the similarity between tasks. The best part of this paper was that they investigated extensively about "*Task Transferability*" (26 NLP tasks and 160 combinations of source-target tasks) based on "Task similarity" (cosine similarity) which helps to identify which set of source tasks are likely to yield positive results on a novel target task.


### Visualizing the Problem

Up until this, the road has been set for the IBM researchers, as earlier papers have extensively tried parameter-efficient approaches. We have a set of source tasks  $\mathbf{\textit{S}} = {S_{1}, S_{2}, ... , S_{k}}$ and a set of target tasks  $\textit{T} = {T_{1}, T_{2}, ... , T_{k}}$.
Now, they want to tackle the **Multitask challenge**, by killing two birds with a stone. They are going to implement 
+ Task transferability(Vu et al., 2021)
+ Compact model/prompt matrix by Knowledge distillation

via. building a single prompt matrix
+ commonalities between source tasks
+ specialities of a given task

### Current Approach- Dubbed MPT
In the previous approaches, soft prompts are individually learned per task using various approaches, particularly vanilla prompt tuning (see Vanilla prompt tuning). This phase of the process is called source training. In source training, we aggregate the prompts we learned from the set of source tasks. In the next phase, target adaptation, we adapt a prompt from the pool of aggregated pre-trained (on source tasks) prompts and initialize the prompt for further fine-tuning on a target task based on a (potentially learned) similarity measure.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226112617-20eb9806-b5b2-4555-9c1a-12829eba696d.png">
  <em>Figure 2.An illustration on prompt decomposition for two tasks.</em>
</p>

In the dubbed version of Multitask prompt tuning (MPT), we follow the same strategy of Source training and target adaptation with a slight twist. It seems that simply sharing the aggregated soft prompts in the Vanilla version did not fit the target task so well and resulted in a gap in the performance of the PLM w.r.t Source and Target tasks. 
#### Source Training:
The primary goal of the authors was to design a single shared prompt matrix $\widehat{P^*}$ that could easily adapt to the nature of the target tasks. They try to model this single prompt matrix as a combination of two superpowers. First one, it should carry all the common flavors from the source tasks (RTE, QA, Text summarization, etc.), and second, it should be good at a specific task(say, for a QA task). How do they learn this single shared prompt matrix, is the "**big question**"?
Well, they literally "*learn*" this prompt from a very popular and common concept in Machine Learning which is *Knowledge Distillation*. If we are talking about KD, we must have Teachers and Students! 

We already have a bunch of soft prompts from the source tasks, and these soft prompts per task will be the teachers. Each Teacher soft prompt is an ‘expert’ on the task it was pre-trained upon. This knowledge has to be distilled, compressed, and delivered to the student prompt. For a task *k* in the source task set, we have a teacher prompt $P_{k}$ and a student prompt $\widehat{P_k}$. The student prompt is expected to get the gist from all the teacher prompts (single shared matrix, our primary goal to learn) and also perform well on the task it was designed for initially(task-specific matrix).

In parallel, these soft prompts will be decomposed into two matrices A and B(Prompt decomposition). We decompose the soft prompt $P_k$ for the k-th task into two parts, as shown in Figure 3. Let $P^{k} \in \mathbb{R}^{lxd}$  denote the shared prompt across all tasks, and further let $u^{k} \in \mathbb{R}^{l}$, $v^{k} \in \mathbb{R}^{d}$ be the task-specific vectors for each task $\it{k}$. The task-specific vectors form a rank-one matrix $W_{k} = u_{k} \otimes v_{k}^{T}$ , which has the same dimensions as the shared prompt $P^{\ast}$ . The task prompt $\hat{P}$ 
for *k*-th source task is then parameterized as:

<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226111397-72205dba-d768-4152-a7d0-d12f84cf31b2.gif" alt="Equation 1"/>
</p>

General information across the set of source tasks ***S*** can be captured by “slow” weights $P^{*}$ shared across tasks, while the “fast” weights $W_{k}$ could then encode task-specific knowledge for $S_{k}$ in a low-rank subspace.

#### Losses
The first loss equation, basically aims to minimize the KL divergence between shared prompt matrix $P^{\ast}$ and task-specific parameters $u_{k}$ and  $v_{k}$

<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226167093-22eb97b8-ffc6-4e4e-8017-8c29b8d78db2.JPG" alt="Equation 3"/>
</p>

This tries to bring the output probability distributions of the teachers and student prompts closer to each other. The next distillation loss, is computed by mean squared loss between teacher and student networks hidden states.
<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226167119-3685ff27-6c8c-4672-9485-1b3c74c2d31f.JPG" alt="Equation 4"/>
</p>

And finally, the total loss is computed as following,
<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226167151-eba04322-e0f9-486d-ac71-d5743b7d1972.JPG" alt="Equation 4"/>
</p>

#### Target Adaptation
Now that we have learnt the shared common prompt matrix $P^{\ast}$ we can now initialize the target prompt for task $T_{t}$ to be as follows:
$\widehat{P_t} = P^{\ast} \circ (u_{k} \otimes v_{k}^{T})$
and optimize with the regular task loss as $L_{PLM} = - \sum_{i}^{}log P(y_{i}|x_{i} ; \theta , \mathbf{P} )$

##### Baselines 
MPT was compared against the following baselines: (1) Full finetuning (FT), where all the model parameters are tuned during adaptation on each downstream task. (2) Vanilla prompt tuning (PT) (Lester et al., 2021), where target prompt vectors are initialized by randomly sampled top vocabularies. (3) Existing prompt transfer methods, including SPoT (Vu et al., 2022) and ATTEMPT (Asai et al., 2022), which initialize target prompts by retrieving or aggregating source prompts. (4) Popular parameter-efficient methods including Adapters (Houlsby et al., 2019) and
BitFit (Zaken et al., 2022).

**Implementation details** : For source training part, MPT was trained on a mixture of tasks. For target adaptation part, they reused the shared prompt matrix and take average of the *source* task specific vectors to initialize the target specific vector.

#### Results and Discussion
+ Parameter efficient : The total number of tunable parameters for a single target task is $(l × d) + (l + d)$. After training, this can further be compressed into a single matrix of size $l × d^{2}$.


MPT was judged on four benchmarks and evaluated on each task following the baselines.
+ MPT establishes new state-of-the-art results for parameter-efficient finetuning on both GLUE and SuperGLUE
+ If we talk about parameter efficient, ADAPTERS is the most competetive and accurate of them all. But,  MPT is far more parameter efficient and requires 4× fewer
task-specific parameters. Not just that, MPT also beats full-finetuning on GLUE and SUPERGLUE despite using just 0.035% as many task-specific parameters.
+ There is significant gap in full finetuning and MPT on the MRQA benchmark, which brings us to question the accuracy of MPT in this direction.
+ Few-shot adaptation : MPT can effectively use cross-task
knowledge in source tasks to target tasks where there are only a few labeled examples.
+ *Can MPT transfer knowledge from NLU task to NLG task?*
They conduct a series of experiments to test whether prompt decomposition learned from source NLU tasks can generalize to target NLG task. The T5-LARGE prompt trained by six diverse source NLU tasks was tested on two NLG tasks, E2E (Novikova et al., 2017) and WebNLG (Gardent et al., 2017).  BLEU
improvements over PT are 3.03% and 6.25% on E2E and WebNLG tasks respectively, showing the effectiveness of our approach on both NLU (e.g., classification, NLI, QA tasks) and NLG tasks.
+ Ablation w.r.t. decomposition and distillation
To establish the importance of decomposition and distillation they carried out an ablation study on SUPERGLUE, which demonstrates that the shared component can effectively capture the rich cross-task knowledge that is beneficial for target downstream tasks.
<p align="center">
  <img src="https://user-images.githubusercontent.com/39300414/226168009-16699c34-ab09-4bbc-b9a1-f4422bd61d33.JPG">
  <em>Figure 3. Ablation results on prompt decomposition and distillation</em>
</p>





