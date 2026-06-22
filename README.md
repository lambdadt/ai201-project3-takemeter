# TakeMeter: Discourse Quality Classifier for r/LocalLLaMA

A fine-tuned text classifier that evaluates post quality on r/LocalLLaMA, a Reddit community for discussing locally hostable AI models. The classifier assigns each post to one of three labels: **High-Signal Technical**, **Low-Effort Noise**, or **News & Meta Discourse**. The goal is to serve as a filtering mechanism that separates substantive content from the noise that increasingly fills the subreddit feed.

---

## Community & Label Taxonomy

r/LocalLLaMA is a hub for discussing locally hostable large language models. The discourse ranges from PhD-level deep dives on quantization and multi-GPU memory bandwidth to absolute beginners asking the same basic questions daily. This extreme variance in effort and technical depth makes it an ideal candidate for a classifier aimed at filtering out noise to reclaim a high-signal environment.

### Labels

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|---|
| **High-Signal Technical** | Posts with novel technical value — detailed project showcases, specific hardware benchmarking/troubleshooting with clear prior research, or in-depth technical knowledge sharing. | "Why is AutoRound being slept on so hard? I've been experimenting with it on Qwen3.6 27B (AMD setup), and the perplexity/accuracy retention at low bits blows standard AWQ or RTN out of the water." | "Thinking loop bug in OpenCode with local model? I'm using two GPUs (16GB+12GB, RTX4080 + RTX3080Ti). I tried changing models and quants (Qwen 3.6-27B_Q6_K, Qwen 3.6-27B_Q5_K, Qwen 3.6-27B_Q4_K_M) with no success." |
| **Low-Effort Noise** | Posts that clutter the feed — repetitive beginner questions lacking prior research, vague clickbait, useless social media screenshots, or AI-generated ("vibecoded") projects with no real utility. | "what are the best / most recent open LLMs to come out recently? Hey everybody the last LLM drop I really remember was Gemma 4. What has come out since then and which are the best?" | "Vercel CEO: 'Almost shocked' by how good GLM-5.2 is at coding. Guillermo Rauch says he is 'genuinely impressed, almost shocked...' Source: X post." |
| **News & Meta Discourse** | Legitimate open-source model releases, significant industry news impacting the open-source ecosystem, or high-effort opinion pieces about the state of the community. | "Heretic has been served a legal notice by Meta, Inc. The Heretic Free Software Project was forced to remove Llama derivative models from all repositories." | "DGX Spark is being defamed. Yes, the memory bandwidth isn't amazing. Yes, it has fallen victim to price increases. Fine. With that out of the way, it's a totally cool setup..." |

**Decision rule for ambiguous posts:** The deciding factor is **demonstrated effort and specificity**. If a post includes specific constraints (e.g., exact quantization levels tested, hardware already tried), it leans High-Signal Technical. If truly 50/50, default to Low-Effort Noise to protect feed quality.

### Hardest Edge Cases Identified

1. **Niche "what model?" questions with ethical/specific framing** — e.g., "Models for Psychological Review of Conversations" asks for specific domain models with ethical caveats. Decided: High-Signal Technical because the problem domain is concrete and non-trivial.
2. **High-effort packaging of a repetitive question** — e.g., "Can I realistically get close to Claude/Codex capabilities locally?" is detailed with hardware specs and budget, but the core question has been asked hundreds of times. Decided: High-Signal Technical because the constraints are non-standard.
3. **Relevant news in a Twitter-screencap format** — e.g., "More Gemma 4 models incoming" is just a link. Decided: Low-Effort Noise because the format adds nothing beyond the tweet.
4. **Quantitative claims with clickbaity presentation** — e.g., "Token speed doubled + kv cache now need low vram" has specific numbers and a GitHub repo but is framed as hype. Decided: Low-Effort Noise because the unverifiable presentation undermines the signal.

For full documentation of label definitions, decision rules, and edge cases, see [planning.md](planning.md).

---

## Data Collection

**Source:** Posts were collected manually from r/LocalLLaMA, capturing the title and post body for each entry.

**Labeling process:** I used a DSPy pipeline with DeepSeek V4 Pro to classify each post using a 9-category fine-grained taxonomy. These were then manually reviewed and mapped to the 3-label schema. This combined LLM-assisted pre-labeling with human verification — the LLM provided a strong starting point, and I corrected labels where the initial classification disagreed with my reading of the post.

**Dataset size:** 63 annotated posts. During collection, I observed that the subreddit's posts were highly repetitive — the same patterns of beginner questions, Twitter screencap hype, and model release announcements appeared dozens of times with near-identical structure. Rather than padding the dataset with near-duplicates, I prioritized diversity over volume.

**Label distribution:**

| Label | Count | % |
|---|---|---|
| Low-Effort Noise | 28 | 44.4% |
| High-Signal Technical | 18 | 28.6% |
| News & Meta Discourse | 17 | 27.0% |

All labels exceed the 20% minimum threshold. The Low-Effort Noise dominance mirrors the natural distribution of the subreddit.

**Three examples I found genuinely difficult to label:**

1. **"Models for Psychological Review of Conversations"** — straddles High-Signal Technical and Low-Effort Noise. The post frames a specific non-trivial use case with ethical considerations, but never names a model the OP has tried. Labeled High-Signal Technical because the concrete problem domain elevates it above a generic "what model?" question.

2. **"Can I realistically get close to Claude/Codex capabilities locally?"** — straddles High-Signal Technical and Low-Effort Noise. Extremely detailed with hardware specs and budget, but asks a question asked hundreds of times. Labeled High-Signal Technical because the non-standard constraints (NVME adapter, specific budget) make it a genuine discussion prompt.

3. **"More Gemma 4 models incoming"** — straddles News & Meta Discourse and Low-Effort Noise. The news (upcoming model releases) is genuinely relevant, but the post is literally a Twitter link with one line of speculation. Labeled Low-Effort Noise because the format adds zero original value.

---

## Model & Training

**Base model:** `distilbert-base-uncased` (HuggingFace), a 66M-parameter distilled version of BERT with a classification head. Chosen because it fits on Colab's free T4 GPU and is the recommended baseline for this project.

**Training approach:** Full fine-tuning (not LoRA/adapters) on an A100 GPU with the HuggingFace `Trainer` API.

### Key Hyperparameter Decisions

| Parameter | Value | Rationale |
|---|---|---|
| `num_train_epochs` | 15 | Default is 3, but with only 44 training examples (~3 steps/epoch), more epochs were needed for the model to converge. Validation accuracy plateaued at epoch 10 (0.667), so the last 5 epochs were unnecessary — 10 would have sufficed. |
| `warmup_steps` | 30 | Added to stabilize early training on the extremely small dataset. However, 30 warmup steps out of ~45 total means most of training was spent warming up — in hindsight, a value of 5-10 would have been more appropriate. |
| `learning_rate` | 2e-5 | Standard for BERT-family fine-tuning. |
| `per_device_train_batch_size` | 16 | Default; GPU memory was not a constraint. |

**Data split:** 70% train (44 examples), 15% validation (9), 15% test (10) — stratified to maintain label proportions across splits.

**Training trajectory:** Validation loss dropped from 1.09 to 0.81. Validation accuracy rose from 0.33 (epoch 1) to a ceiling of 0.67 (epoch 10), where it stayed for the remaining 5 epochs.

---

## Evaluation Results

### Test Set Composition

| Label | Test Examples |
|---|---|
| Low-Effort Noise | 5 |
| News & Meta Discourse | 3 |
| High-Signal Technical | 2 |

**Important caveat:** The test set has only 10 examples, so a single misclassification shifts accuracy by 10 percentage points. Per-class metrics for classes with 2-3 examples are especially noisy. All results below should be interpreted with this caveat.

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq `llama-3.3-70b-versatile`) | 0.600 |
| Fine-tuned DistilBERT | 0.700 |
| **Fine-tuning improvement** | **+0.100** |

The fine-tuned model outperforms the zero-shot baseline by 10 percentage points. However, this improvement represents exactly 1 additional correct classification on a 10-example test set, so it is not statistically significant.

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| News & Meta Discourse | 0.00 | 0.00 | 0.00 | 3 |
| Low-Effort Noise | 0.71 | 1.00 | 0.83 | 5 |
| High-Signal Technical | 0.67 | 1.00 | 0.80 | 2 |
| **Macro Avg** | 0.46 | 0.67 | 0.54 | 10 |
| **Weighted Avg** | 0.49 | 0.70 | 0.58 | 10 |

### Per-Class Metrics — Baseline Model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| News & Meta Discourse | 0.50 | 0.33 | 0.40 | 3 |
| Low-Effort Noise | 1.00 | 0.60 | 0.75 | 5 |
| High-Signal Technical | 0.40 | 1.00 | 0.57 | 2 |
| **Macro Avg** | 0.63 | 0.64 | 0.57 | 10 |
| **Weighted Avg** | 0.73 | 0.60 | 0.61 | 10 |

### Confusion Matrix — Fine-Tuned Model

| | Predicted N&M | Predicted LEN | Predicted HST |
|---|---|---|---|
| **True: News & Meta (3)** | **0** | 2 | 1 |
| **True: Low-Effort Noise (5)** | 0 | **5** | 0 |
| **True: High-Signal Technical (2)** | 0 | 0 | **2** |

Key: N&M = News & Meta Discourse, LEN = Low-Effort Noise, HST = High-Signal Technical. Correct predictions are on the diagonal in **bold**.

Every error is in the News & Meta Discourse row — the model correctly identifies all High-Signal and Low-Effort posts but cannot distinguish News & Meta Discourse from either category.

### Sample Classifications

Five test-set posts run through the fine-tuned model, each shown with the predicted label and softmax confidence score:

| Post Snippet | True Label | Predicted Label | Confidence |
|---|---|---|---|
| "what are the best / most recent open LLMs to come out recently? Hey everybody the last LLM drop I really remember was Gemma 4..." | Low-Effort Noise | Low-Effort Noise | ~0.79 |
| "Thinking loop bug in OpenCode with local model? For some reason Opencode keeps self-prompting itself... I'm using two GPUs (16GB+12GB, RTX4080 + RTX3080Ti)... I tried changing models and quants..." | High-Signal Technical | High-Signal Technical | ~0.82 |
| "Gemma 4 with quantization-aware training... blog.google... Google's collections: [HuggingFace links]... Unsloth's analysis..." | News & Meta Discourse | Low-Effort Noise | 0.48 |
| "There has to be a way to avoid retraining entire base model... MoE... LoRA... MLP layers... speculative decoding..." | News & Meta Discourse | High-Signal Technical | 0.38 |
| "DGX Spark is being defamed. Yes, the memory bandwidth isn't amazing... With two though... DS4 Flash local at 1800 prefill and 40 gen..." | News & Meta Discourse | Low-Effort Noise | 0.40 |

**Correct prediction spotlight** (row 1): The model classified the "best open LLMs" query as Low-Effort Noise with high confidence (~0.79). This is reasonable because the post matches every pattern in the Low-Effort Noise definition: it's a vague "what model should I use?" question, demonstrates zero prior research, and is easily answered by searching the subreddit. The model correctly recognized the absence of hardware specifics, debugging steps, or technical framing that would have elevated it to High-Signal Technical.

**Correct prediction spotlight** (row 2): The "thinking loop bug" post was classified as High-Signal Technical with high confidence (~0.82). This is reasonable because the post includes: a specific bug description, multiple models and quantization levels tried (Q6_K, Q5_K, Q4_K_M), two inference backends tested (llama.cpp and LMStudio), full launch commands, and config files. The model learned to associate diagnostic thoroughness + configuration details with High-Signal Technical.

**Confidence pattern:** The three wrong predictions all have low confidence (0.38–0.48) — the model was unsure about these, which suggests it recognized they didn't fit its learned patterns but lacked the third-class signal to place them correctly. The correct predictions have substantially higher confidence, suggesting the model has well-formed decision boundaries for the two classes it did learn.

---

## Error Analysis

All three mistakes are on the **News & Meta Discourse** class. The model achieved 0% recall and 0% precision for this label — functionally, it learned nothing about what distinguishes this category.

### Error 1: "Gemma 4 with quantization-aware training"

- **True:** News & Meta Discourse → **Predicted:** Low-Effort Noise (confidence: 0.48)
- **Why it failed:** This post announces Google's QAT-enhanced Gemma 4 release with multiple HuggingFace links and Reddit metadata (upvote counts). The model appears to have learned a surface-level pattern: posts with many URLs and Reddit metadata → Low-Effort Noise. The semantic signal ("this is a significant model release announcement") is lost because the structural form (link-heavy, metadata-cluttered) overpowers it. This is a **format-content mismatch**: the post is newsworthy but looks structurally like low-effort spam.
- **Directional pattern:** News & Meta → Low-Effort Noise accounts for 2 of the 3 errors, suggesting the model defaults to the majority class when it encounters posts that don't match High-Signal Technical vocabulary.

### Error 2: "There has to be a way to avoid retraining entire base model for adding latest information to it"

- **True:** News & Meta Discourse → **Predicted:** High-Signal Technical (confidence: 0.38)
- **Why it failed:** This post discusses a speculative architecture idea (kernel/userspace model split) using technical terms: "MoE," "LoRA," "MLP layers," "speculative decoding." The model keys on technical vocabulary and defaults to High-Signal Technical. But the post is actually an opinion/speculation piece — it proposes an idea without implementation details, benchmarks, or evidence. This is the hardest type of boundary: the post *sounds* technical, but the substance is opinion.
- **Directional pattern:** News & Meta → High-Signal Technical accounts for 1 of the 3 errors. This is the opposite problem from Error 1 — technical vocabulary triggers the High-Signal classifier even when the post lacks substance.
- **Labeling question:** Was this post labeled correctly as News & Meta? Re-reading it, the post is indeed a speculative discussion framed around a metaphor (kernel/userspace), not a technical investigation. It belongs in News & Meta. The model simply cannot distinguish "technical-sounding opinion" from "actual technical content."

### Error 3: "DGX Spark is being defamed"

- **True:** News & Meta Discourse → **Predicted:** Low-Effort Noise (confidence: 0.40)
- **Why it failed:** This is an opinion piece defending the DGX Spark hardware. It discusses pricing, power consumption, and scalability — topics that are common in Low-Effort Noise posts (e.g., "GPU prices are crazy"). The model sees hardware pricing talk and classifies it as noise. But this post is a structured argument with specific numbers and counterpoints — it's a high-effort opinion, which makes it News & Meta Discourse. The model cannot distinguish "hardware complaints" (Low-Effort Noise) from "hardware defenses with reasoning" (News & Meta Discourse) because both share the same vocabulary domain.
- **Directional pattern:** Same as Error 1 — the model defaults to Low-Effort Noise when the topic overlaps with noise vocabulary.

### Systematic Pattern Across All Errors

The unifying pattern is that **News & Meta Discourse has no distinctive lexical fingerprint**. A model release announcement (Error 1) uses the same URLs as a low-effort link share. A technical-sounding opinion (Error 2) uses the same vocabulary as a genuine technical deep dive. A hardware opinion piece (Error 3) uses the same pricing numbers as a low-effort complaint about GPU prices. The class is defined by *intent and context* (is this a news event? is this a reasoned opinion?), not by vocabulary — and a 66M-parameter model cannot learn intent from 12 training examples.

---

## Baseline Comparison

The zero-shot baseline uses Groq's `llama-3.3-70b-versatile` with a system prompt that defines each label using a one-sentence definition and a community-relevant example post, then instructs the model to "respond with ONLY the label name." The same 10 test-set posts were sent individually with `temperature=0` and `max_tokens=20`. The prompt was written using the label definitions and examples from [Section 2 of planning.md](planning.md#2-label-definitions).

The baseline achieved 0.600 accuracy vs. the fine-tuned model's 0.700. The baseline had a more balanced error profile — it got at least one News & Meta post correct (recall: 0.33) and spread errors across both Low-Effort Noise (recall: 0.60) and News & Meta. The fine-tuned model concentrated all errors on News & Meta, suggesting fine-tuning sharpened the High-Signal vs. Low-Effort boundary at the expense of losing the third class entirely.

The 70B-parameter baseline's more balanced performance on a task where a 66M-parameter fine-tuned model collapsed on one class suggests that **News & Meta Discourse requires reasoning about context and intent** — something that parameter scale enables but that the small training set could not teach.

---

## Reflection: What the Model Learned vs. What I Intended

### Intended Decision Boundary (from label definitions)

The three labels were designed around **intent and context**:
- **High-Signal Technical:** Does this post contribute novel, verifiable technical value? (e.g., benchmarks, build logs, reproducible debugging)
- **Low-Effort Noise:** Does this post lack effort or novelty? (e.g., no prior research, clickbait, social media reposts)
- **News & Meta Discourse:** Is this post an event the community should know about, or a reasoned opinion about the ecosystem? (e.g., model releases, community meta-discussion)

The intention was that High-Signal Technical and Low-Effort Noise would be distinguished primarily by *demonstrated effort*, and News & Meta Discourse would capture posts that are valuable in a different way — not technical, but still worth keeping.

### Actual Decision Boundary (what the model learned)

The model learned a **lexical-proxy boundary**, not an intent boundary:

- **URL density → Low-Effort Noise.** Posts with many URLs and Reddit metadata are classified as noise regardless of content. This works for actual spam but also catches legitimate model release announcements (Error 1).
- **Technical vocabulary → High-Signal Technical.** Posts containing terms like "LoRA," "MLP layers," "KV cache," "benchmark" are classified as High-Signal Technical even if they are speculative opinion (Error 2).
- **Hardware pricing talk → Low-Effort Noise.** Posts discussing GPU prices, power consumption, or hardware cost are classified as noise regardless of whether the argument is structured (Error 3).
- **News & Meta Discourse → absorbed by nearest neighbor.** The model cannot represent this class at all — every News & Meta post is pushed to whichever binary class its vocabulary most resembles.

### The Gap

The gap is between **what the labels mean** (the intent behind a post) and **what the labels look like** (the vocabulary a post uses). My label definitions say: judge whether this person did research, whether this announcement matters to the community, whether this argument is reasoned. The model can only see: does this text contain URLs? Technical jargon? Price numbers?

This is not purely a data volume problem. It's also a **label design tension**: News & Meta Discourse is defined as "significant industry news OR high-effort community opinion." A model release (news) and a community rant (opinion) have almost nothing in common lexically, but they share a label. The model has to learn that "Gemma 4 released" and "DGX Spark is being defamed" belong together, which requires understanding that both are *contextually relevant* — a concept that 12 training examples and 66M parameters cannot encode.

**What the model overfit to:**
- Reddit metadata artifacts (upvote counts, "r/LocalLLaMA —" prefixes, page numbers) as noise markers
- Code blocks and CLI commands as High-Signal markers
- The majority class (Low-Effort Noise) as the default prediction for anything ambiguous

**What the model missed:**
- The concept of "newsworthiness" — that a link-heavy post can still be valuable if it announces a model release
- The distinction between "technical vocabulary used for argument" (News & Meta) and "technical vocabulary used for investigation" (High-Signal Technical)
- The News & Meta Discourse class entirely — it has no representation in the model's feature space

### Three Things I Would Change

1. **More News & Meta Discourse examples.** With only 12 training examples for a class defined by context rather than vocabulary, the model had no chance. Doubling this class would be the highest-impact change.

2. **Narrower label definition for News & Meta Discourse.** Currently this label bundles model releases (concrete, fact-based) and community opinion pieces (subjective, argument-based) into one category. These have very different lexical profiles. Splitting them or tightening the definition to exclude opinion pieces would make the boundary more learnable.

3. **More explicit boundary examples in the training data.** None of the 12 News & Meta training examples were specifically chosen to illustrate the difference between "hardware discussion as news/opinion" and "hardware discussion as noise." Adding 3-5 examples that sit at this boundary, with deliberate labeling, would teach the model where the line is.

**What went right:** The binary High-Signal vs. Low-Effort distinction works. The model correctly identifies all High-Signal Technical posts (recall 1.00) and all Low-Effort Noise posts (recall 1.00, precision 0.71). For a filtering tool where the primary goal is surfacing technical content, this is a functional starting point — the model fails in a predictable, limited way (suppressing News & Meta rather than mixing it with high-signal content).

---

## Limitations

- **Small test set (10 examples)** makes all metrics unreliable at the individual-class level. A single prediction change shifts accuracy by 10%.
- **63 training examples** is well below the recommended 200 for fine-tuning. The model's inability to learn News & Meta Discourse is almost certainly a data volume problem rather than a fundamental limitation of DistilBERT.
- **LLM-assisted labeling** means the ground truth labels are not purely human-generated. While I manually reviewed every label, some boundary cases may reflect the LLM's interpretations rather than consistent human judgment.
- **Single-community focus.** The model is trained exclusively on r/LocalLLaMA and would not generalize to other AI communities with different discourse norms.
- **Temporal specificity.** The posts were collected in June 2026 and the vocabulary is tied to models and events of that period (GLM-5.2, Gemma 4, DGX Spark). The classifier would age quickly as new models and community trends emerge.

---

## Spec Reflection

**One way the spec helped guide implementation:** The requirement to define labels that are **mutually exclusive, grounded in community norms, and precise enough that two readers would agree** forced the most valuable design work in the project. Rather than jumping to "good post vs. bad post," I had to watch myself classify 30+ real posts before committing to labels, which surfaced the News & Meta Discourse category — it wouldn't have existed if I'd designed labels from memory. The edge case documentation requirement (Section 3 of planning.md) also proved essential: when the model later collapsed on News & Meta, I could trace the failure back to the four boundary cases I'd documented, each of which previewed the exact failure mode (format-content mismatch, technical vocabulary in opinion posts, hardware pricing as a confounding signal).

**One way my implementation diverged from the spec and why:** The spec calls for a dataset of at least 200 annotated examples. I stopped at 63. This was a deliberate choice made after reviewing ~40-50 additional candidate posts and finding them to be near-duplicates of the examples I'd already collected. The subreddit's post patterns are genuinely repetitive — the same beginner questions, Twitter screencaps, and model release announcements appear with near-identical structure dozens of times. Padding the dataset to 200 with these would have inflated accuracy on the test set (more examples of the same distribution = better fit) without improving the model's ability to generalize to the genuinely novel posts that make a filter useful. However, the cost is real: a 10-example test set makes all per-class metrics noisy, and the model's inability to learn News & Meta Discourse may be partially attributable to insufficient training examples (only 12) rather than purely a label design problem. In hindsight, I would have specifically hunted for diverse News & Meta Discourse posts to bring that class to at least 20 training examples while keeping total dataset size manageable.

---

## AI Usage Disclosure

### Instance 1: Initial Category Generation (Gemini)

**What I directed the AI to do:** After manually reading and commenting on ~60 posts from r/LocalLLaMA, I had a collection of posts with my own short commentary (e.g., "low effort," "interesting tech insight," "vibecoded slop") but couldn't crystallize them into a concrete set of categories. I compiled all 60 posts with my commentary into a single prompt and asked Gemini to "identify a list of unique categories to which I can assign each of these posts."

**What the AI produced:** A taxonomy of 9 categories:
1. Projects & Showcases
2. Technical Deep Dives & Hardware
3. Relevant News & Releases
4. Meta & General Discussions
5. High-Effort Technical Inquiries
6. Beginner & Repetitive Support
7. Low-Effort Slop & Hype
8. Other (Good)
9. Other (Bad)

**What I changed:** I accepted the 9-category taxonomy as-is for the initial DSPy classification pipeline. However, when designing the 3-label taxonomy for fine-tuning, I collapsed these 9 categories into 3, making my own judgment calls about which categories belonged together — specifically, I merged category 4 (Meta & General Discussions) with category 3 (Relevant News & Releases) into "News & Meta Discourse," which I later identified as an overly broad grouping. I also dropped the "Other" categories entirely, requiring every post to fit one of the three labels.

### Instance 2: Automated Labeling with DSPy (DeepSeek V4 Pro)

**What I directed the AI to do:** I wrote a DSPy pipeline (`classify_posts.py`) using DeepSeek V4 Pro via a ChainOfThought module to classify all 63 posts into the 9-category taxonomy. The pipeline received the full post content and my human commentary for each post, and produced a category number, category name, and reasoning for each classification. The output was saved to `classification_results.json`.

**What the AI produced:** The initial output (`classification_results.json`) contained DeepSeek's 9-category classification for all 63 posts, with detailed reasoning chains for each decision.

**What I changed and overrode:** I manually reviewed every single AI-assigned label. In several cases, my human commentary disagreed with the AI's classification:

- **Post: "Models for Psychological Review of Conversations"** — the AI classified this as "High-Effort Technical Inquiries" despite my commentary saying "Low effort. OP should have done more research before posting this." I kept the AI's classification because on re-reading, the post's concrete domain framing (psychological analysis, ethical considerations) justified it above a generic "what model?" question.

- **Post: "Best image vision model runnable on RTX 6000 Pro"** — the AI classified this as "High-Effort Technical Inquiries" despite my commentary saying "Kind of low effort because the OP could have done some research before this." I agreed with the AI here because the post mentions having already tried Gemma 4 31B and comparing it to Qwen 3.6, which demonstrates prior research.

- **Post: "I got tired of runaway local agent loops... eBPF circuit breaker"** — the AI classified this as "Low-Effort Slop & Hype," agreeing with my commentary that it was "likely some vibecoded AI slop." I kept this classification. However, this was a case where the post's content alone (detailed technical architecture, eBPF, SimHash, OLS regression) would have looked like a legitimate project showcase — the AI's ability to incorporate my commentary was essential.

The manually-reviewed and corrected output was saved as `classification_results_final.json`, which was then mapped to the 3-label taxonomy for the `labeled_posts.csv` training dataset.

### Instance 3: Error Pattern Analysis (Claude)

**What I directed the AI to do:** After fine-tuning, I provided all three misclassified test examples plus their true and predicted labels to Claude and asked it to identify common themes or structural patterns across the failures.

**What the AI produced:** The AI identified that all three errors involved the News & Meta Discourse class being misclassified due to surface-level lexical proxies: URL-heavy posts being treated as noise, technical vocabulary triggering a High-Signal classification, and hardware pricing discussion collapsing into Low-Effort Noise. It hypothesized that the class lacks a "distinctive lexical fingerprint."

**What I changed:** I verified this hypothesis by re-reading all three posts and confirmed the pattern. The AI's framing ("lexical proxy" vs. "intent boundary") directly shaped the Reflection section's structure. I did not discard any part of the AI's analysis because it matched what I observed in the confusion matrix: all three errors were News & Meta → something else, and the directional split (2× to Low-Effort, 1× to High-Signal) aligned with which vocabulary each post most resembled.

---

## Files in This Repository

| File | Description |
|---|---|
| `planning.md` | Full design documentation: label taxonomy, edge cases, data collection plan, evaluation metrics reasoning, AI tool plan, training decisions, and post-training reflection |
| `labeled_posts.csv` | 63 annotated posts with `text` and `label` columns |
| `ai201_project3_takemeter_starter_clean.ipynb` | Colab notebook with fine-tuning pipeline, baseline comparison, and evaluation code |
| `evaluation_results.json` | Summary metrics exported from the notebook |
| `confusion_matrix.png` | Confusion matrix visualization for the fine-tuned model |
| `classification_results_final.json` | Raw DSPy classification output (9-category) before mapping to the 3-label taxonomy |
