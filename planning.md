Here is the `planning.md` document structured to directly address the requirements of the CodePath assignment, drawing upon the analysis of the `r/LocalLLaMA` community.

---

# Project Planning: r/LocalLLaMA Post Quality Classifier

## 1. Community Context

**Target Community:** `r/LocalLLaMA`
**Why this community?** `r/LocalLLaMA` is a hub for discussing locally hostable AI models. As open-source AI has exploded in popularity, the subreddit has experienced massive growth.
**Why is it a good fit for classification?** The discourse is incredibly varied, ranging from highly technical PhD-level deep dives on quantization and multi-GPU memory bandwidth to absolute beginners asking the exact same basic questions daily. This extreme variance in effort and technical depth makes it an ideal candidate for a text classifier aimed at filtering out "noise" to reclaim a high-signal environment for advanced users.

## 2. Label Definitions

To keep the classifier highly effective and bounded, the taxonomy has been consolidated into **three** distinct labels:

* **High-Signal Technical:** Posts that provide novel technical value, detailed project showcases, functional coding harnesses, or specific hardware benchmarking/troubleshooting with clear prior research.
* *Example 1:* "Thinking loop bug in OpenCode with local model? For some reason Opencode keeps 'self-prompting' itself... I'm using two GPUs (16GB+12GB, RTX4080 + RTX3080Ti)..."
* *Example 2:* "Why is AutoRound being slept on so hard? I’ve been experimenting with it on Qwen3.6 27B lately (running an AMD setup), and the perplexity/accuracy retention at low bits absolutely blows standard AWQ or RTN out of the water."


* **Low-Effort Noise:** Posts that clutter the feed, including repetitive beginner questions lacking prior research, vague clickbait, useless social media screenshots, or "vibecoded" AI slop projects with no real utility.
* *Example 1:* "what are the best / most recent open LLMs to come out recently? Hey everybody the last LLM drop I really remember was Gemma 4... what has come out since about then?"
* *Example 2:* "Vercel CEO: 'Almost shocked' by how good GLM-5.2 is at coding. Guillermo Rauch says he is 'genuinely impressed, almost shocked...'"


* **News & Meta Discourse:** Legitimate open-source model releases, significant industry news directly impacting the open-source ecosystem, or high-effort opinion pieces/discussions about the state of the community.
* *Example 1:* "Heretic has been served a legal notice by Meta, Inc. The individual behind the Heretic Free Software Project has been served a notice..."
* *Example 2:* "DGX Spark is being defamed. Yes, the memory bandwidth isn't amazing. Yes, it has fallen victim to price increases. Fine. With that out of the way, it's a totally cool setup..."


## 3. Hard Edge Cases & Ambiguity

**The Core Ambiguity:** The hardest recurring edge case is distinguishing between a *High-Signal Technical* troubleshooting question and a *Low-Effort Noise* beginner question. A secondary boundary exists between *News & Meta Discourse* and *Low-Effort Noise* for posts that announce something relevant but in a low-effort format (e.g., Twitter screenshots).

**Decision Rule:** During annotation, the deciding factor is **demonstrated effort and specificity**. If a question includes specific constraints (e.g., exact PCIE slot limitations, quantization levels tested, hardware already tried), it leans *High-Signal Technical*. If it is a generic "What GPU/model should I buy/use?", it is *Low-Effort Noise*. If an edge case is truly 50/50, default to *Low-Effort Noise* to protect the high-quality feed.

### Specific Difficult Examples

**Example 1 — "Models for Psychological Review of Conversations"**
- **Content:** Asks for model recommendations for psychological analysis of conversation transcripts, notes ethical considerations and informed consent, specifies preference for smaller/accessible models, and frames it as "experimentation in conjunction with trained professionals."
- **Labels considered:** High-Signal Technical vs. Low-Effort Noise
- **Why it was difficult:** The post frames a specific, non-trivial use case and acknowledges ethical concerns, which signals effort. However, it fails to name any models the OP has already tried and is fundamentally a "what model should I use?" question — the exact pattern that defines Low-Effort Noise.
- **Decision:** **High-Signal Technical.** The specificity of the use case (psychological analysis, not generic chat), ethical framing, and hardware-awareness push it above the typical repetitive beginner question. A generic "what's the best local LLM for therapy?" would be Low-Effort Noise — what saves this one is the concrete problem domain.

**Example 2 — "Can I realistically get close to Claude/Codex capabilities locally?"**
- **Content:** A software engineer details their setup (5070 Ti + 5060 Ti, 32GB VRAM), explains their use case (large codebases, long sessions, Opus 4.8 with 1M context), states a budget ($3.5K), and asks for model and hardware upgrade advice.
- **Labels considered:** High-Signal Technical vs. Low-Effort Noise
- **Why it was difficult:** This is the poster child for "high-effort packaging of a fundamentally repetitive question." The post is long, detailed, and well-structured — it looks High-Signal. But "can local models match Claude today?" has been asked hundreds of times on r/LocalLLaMA, and the answer hasn't changed. The presentation is high-effort; the underlying question is low-signal.
- **Decision:** **High-Signal Technical.** The level of detail (exact GPU models, NVME adapter constraints, budget ceiling, specific comparison baseline) transforms the post from a FAQ into a genuine discussion prompt because the constraints are non-standard. A post that just said "can I match Claude locally?" without hardware specifics would be Low-Effort Noise.

**Example 3 — "More Gemma 4 models incoming"**
- **Content:** A single link to an X/Twitter post, with one line of speculation: "possibly the 120B model."
- **Labels considered:** News & Meta Discourse vs. Low-Effort Noise
- **Why it was difficult:** The underlying news (upcoming Gemma 4 variant releases) is genuinely relevant to the community. If the same information were presented in a blog post or official announcement, it would clearly be News & Meta Discourse. But the post itself is a Twitter screencap with zero original analysis — it adds nothing.
- **Decision:** **Low-Effort Noise.** The format matters. A post that is nothing but a social media link with one line of speculation does not meet the bar for "high-effort" news curation. If the author had included context (e.g., "this is what the 120B model's benchmarks might look like based on the 31B version"), it would cross into News & Meta Discourse.

**Example 4 — "This is amazing. Token speed doubled + kv cache now need low vram"**
- **Content:** Claims a breakthrough on Qwen3.6-27B: 38.6 tok/s on a single RTX 3090, KV cache reduced from 21GB to 17.5GB, 36/36 benchmark accuracy maintained. Links to a YouTube demo and a GitHub repo.
- **Labels considered:** High-Signal Technical vs. Low-Effort Noise
- **Why it was difficult:** The post presents specific quantitative claims, a GitHub repository, and a video demo — surface-level markers of a genuine technical contribution. But the title is pure clickbait ("This is amazing"), the claims are not peer-reviewed or independently verified, and the human annotator noted the content reads as "sketchy" and that other commenters agreed.
- **Decision:** **Low-Effort Noise.** The deciding factor is reproducibility and proportionality. A legitimate technical post would frame results as "I tested X and observed Y, here's the methodology" rather than "This is amazing. Token speed doubled." The claims may be real, but the presentation follows the pattern of hype-driven, unverifiable benchmark posts. If the same results were published with a proper methodology section and caveats, it would be High-Signal Technical.


## 4. Data Collection Plan

* **Sourcing:** I collected examples directly from the `r/LocalLLaMA` subreddit by manually browsing and scraping posts, pulling the title and post body.
* **Initial Distribution Goal:** I originally aimed for an even split of roughly 75 examples per label to reach a total dataset of ~225 manually annotated posts.

### Final Dataset: 63 Posts (DSPy-Pre-Labeled)

During collection, I found the posts on `r/LocalLLaMA` to be highly repetitive — the same patterns of beginner questions ("what model for my GPU?"), Twitter screencap hype, and model release announcements appeared dozens of times with near-identical structure. Rather than padding the dataset with near-duplicates, I decided to stop at 63 posts that represented genuinely distinct examples of each label. This deliberately prioritizes diversity over volume, at the cost of a smaller training set.

The tradeoff is real: with 63 posts, the train/validation/test split (70/15/15) yields only 44 training examples, 9 validation, and 10 test — the 10-example test set makes individual metrics noisy. However, a dataset of 200 posts with 80% near-duplicate content would inflate accuracy without testing the model's ability to generalize to actually novel posts.

| 3-Label | Count | % | Mapped From |
|---|---|---|---|
| High-Signal Technical | 18 | 28.6% | Projects & Showcases (4), Technical Deep Dives & Hardware (8), High-Effort Technical Inquiries (6) |
| Low-Effort Noise | 28 | 44.4% | Beginner & Repetitive Support (7), Low-Effort Slop & Hype (21) |
| News & Meta Discourse | 17 | 27.0% | Relevant News & Releases (8), Meta & General Discussions (9) |

All three labels exceed the 20% minimum. Low-Effort Noise is the largest class at 44.4%, which mirrors the natural distribution of the subreddit. News & Meta Discourse, despite being the second-largest at 17 examples, would prove to be the hardest class for the model to learn (0% recall in final evaluation).

**Post-hoc note on annotation:** The 9-category DSPy classifications were mapped to the 3-label taxonomy as follows:
- High-Signal Technical ← Projects & Showcases, Technical Deep Dives & Hardware, High-Effort Technical Inquiries
- Low-Effort Noise ← Beginner & Repetitive Support, Low-Effort Slop & Hype
- News & Meta Discourse ← Relevant News & Releases, Meta & General Discussions


## 5. Evaluation Metrics

Accuracy alone will be misleading for this task due to likely class imbalance (e.g., if 70% of the subreddit is currently noise, a model that just guesses "Noise" every time achieves 70% accuracy but is useless).

**Chosen Metrics:**

* **Precision (on High-Signal Technical):** When the model claims a post is High-Signal, how often is it actually High-Signal? We want a clean feed.
* **Recall (on High-Signal Technical):** Out of all the genuinely great posts, how many did the model successfully find? This is the most critical metric. If the classifier aggressively hides amazing deep-dives because it misclassifies them as noise, the tool fails its primary purpose.
* **Macro F1-Score:** To ensure the model is performing competently across all three classes without severely neglecting the *News & Meta* category.

## 6. Definition of Success

For this classifier to be genuinely useful as a community filtering tool (e.g., a browser extension or a custom Reddit feed generator), it needs to act as a trustworthy curator.

**Deployment Standard:** I will consider the model a success if it achieves a **Recall of >0.90** and a **Precision of >0.80** on the *High-Signal Technical* class.

Users are generally forgiving of a few false positives (seeing a low-effort post slip into their curated feed occasionally), but they are highly sensitive to false negatives (missing out on a groundbreaking technical optimization). Therefore, a "good enough" model will err slightly on the side of caution, capturing almost all the great content even if a small amount of noise gets through.

## 7. AI Tool Plan

Since this project focuses heavily on data curation and evaluation rather than writing code, I will strategically use AI tools in three specific phases of the workflow:

* **Label Stress-Testing (Pre-Annotation):** Before I annotate my 200 examples, I will feed my three label definitions (High-Signal Technical, Low-Effort Noise, News & Meta Discourse) and my defined edge cases into an LLM like GPT-4o or Claude 3.5 Sonnet. I will prompt the AI to generate 5–10 realistic, synthetic `r/LocalLLaMA` posts that purposefully sit directly on the boundary between two labels. I will then try to classify these synthetic posts myself. If I cannot cleanly assign them to a single category based on my current definitions, I will revise and tighten the definitions in this document before touching the real data.
* **Annotation Assistance:** I will use an LLM (specifically, Claude 3.5 Sonnet via the DSPy pipeline outlined previously) to pre-label a batch of roughly 100 posts to accelerate the workflow. To ensure transparency and track AI usage, I will add an `AI_Pre_Labeled` boolean column to my dataset tracking sheet (e.g., CSV). I will still manually review, verify, and correct every single AI-generated label before considering the data "ground truth," but the pre-labeling will act as a strong baseline to speed up the process.
* **Failure Analysis (Post-Evaluation):** After evaluating the classifier, I will export a list of all misclassified posts (where the predicted label did not match the ground truth label). I will provide this list to an LLM and prompt it to identify common semantic, structural, or vocabulary patterns among the failures (e.g., "The classifier consistently mislabels short troubleshooting posts as Low-Effort Noise if they lack code blocks"). I will manually verify the AI's hypothesized patterns by reading the raw text of the failed examples to ensure the analysis is accurate before writing up the final evaluation report.

## 8. Training Decisions & Hyperparameters

**Model:** `distilbert-base-uncased` (HuggingFace), fine-tuned on an A100 GPU.

**Hyperparameter decisions:**
| Parameter | Default | Used | Rationale |
|---|---|---|---|
| `num_train_epochs` | 3 | 15 | With only 44 training examples (batch size 16 = ~3 steps/epoch), more epochs were needed for the model to converge. Validation accuracy plateaued at epoch 10, so in hindsight 10 epochs would have sufficed. |
| `warmup_steps` | 0 | 30 | Added to stabilize early training given the extremely small dataset. However, 30 warmup steps on ~45 total steps means 67% of training was spent warming up — likely too conservative. |
| `learning_rate` | 2e-5 | 2e-5 | Standard for BERT-family fine-tuning; kept unchanged. |
| `per_device_train_batch_size` | 16 | 16 | Unchanged; A100 GPU had ample memory. |

**Actual training trajectory:** Validation loss dropped from 1.09 to 0.81 across 15 epochs. Validation accuracy hit 0.667 at epoch 10 and never improved — the last 5 epochs were wasted training time. The model was effectively converged at epoch 10.

## 9. Key Evaluation Findings (Post-Training Reflection)

- **Fine-tuned DistilBERT accuracy: 0.700** vs. **Groq zero-shot baseline: 0.600** — a +0.10 improvement, but on a 10-example test set this delta is within the noise floor of a single prediction swing (1 prediction = 10 percentage points).
- **News & Meta Discourse collapse:** 0.00 precision, 0.00 recall, 0.00 F1. All three test examples of this class were misclassified. The model learns effectively nothing about this label — it functions as a binary classifier between High-Signal Technical and Low-Effort Noise.
- **Root cause:** News & Meta Discourse has no distinctive lexical fingerprint. A news release about a model (e.g., "Gemma 4 QAT released") reads like Low-Effort Noise if it's just a link dump, and a speculative architecture discussion (e.g., "retrain base model") reads like High-Signal Technical if it uses technical vocabulary. The class is defined by *context and intent* — is this a news event? is this a high-effort opinion? — rather than by vocabulary patterns, which a 66M-parameter model with only 12 training examples from this class cannot learn.
- **Success criteria status:** The target was Recall >0.90 and Precision >0.80 on High-Signal Technical. The model achieved Recall 1.00 (both test examples found) and Precision 0.67. However, with only 2 High-Signal test examples, these numbers are not statistically meaningful. A single misclassification (a News & Meta post predicted as High-Signal Technical) was the cause of the precision miss.
- **Labeling insight:** The decision to default ambiguous cases to Low-Effort Noise was intended to protect feed quality. In practice, this meant Low-Effort Noise became the dominant class (44.4%), which may have biased the model toward a conservative classification strategy where it defaults to Low-Effort Noise when uncertain — exactly what we observe with 2 of 3 News & Meta errors going to Low-Effort Noise.
