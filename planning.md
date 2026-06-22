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

**The Ambiguity:** The hardest edge case will be distinguishing between a *High-Signal Technical* troubleshooting question and a *Low-Effort Noise* beginner question. For example: "Upgrading path for RTX Pro 4500 Pro for coding (Qwen3.6-27B) — 1x RTX 5000 or 2x RTX 4500." Is this a basic hardware question, or a niche technical inquiry? Similarly, hype-heavy announcements for genuine model releases blur the line between *Low-Effort Noise* and *News & Meta Discourse*.

**Handling Strategy:** During annotation, the deciding factor will be **demonstrated effort and specificity**. If a question includes specific constraints (e.g., specific PCIE slot limitations, exact quantization levels tested), it will be labeled *High-Signal Technical*. If it is a generic "What GPU should I buy?", it is *Low-Effort Noise*. If an edge case is truly 50/50, I will default to labeling it as *Low-Effort Noise* to strictly protect the high-quality feed.


## 4. Data Collection Plan

* **Sourcing:** I will collect examples directly from the `r/LocalLLaMA` subreddit using the Reddit API (PRAW) or manual scraping, pulling the title and post body.
* **Distribution Goal:** I aim for an even split of roughly 75 examples per label to reach a total dataset of ~225 manually annotated posts.
* **Handling Underrepresentation:** Because *Low-Effort Noise* is highly prevalent, *High-Signal Technical* posts may be underrepresented in a random sample of the "New" feed. If I reach 200 posts and the technical label is starving, I will actively query the subreddit by filtering for specific flairs (e.g., "Resources", "Discussion", "Project") or sort the feed by "Top -> This Month" to deliberately mine positive, high-effort examples to balance the dataset.


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
