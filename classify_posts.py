import dspy
from pathlib import Path
import os
import json

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv


load_dotenv()


category_num_to_name = {
    1: 'Projects & Showcases',
    2: 'Technical Deep Dives & Hardware',
    3: 'Relevant News & Releases',
    4: 'Meta & General Discussions',
    5: 'High-Effort Technical Inquiries',
    6: 'Beginner & Repetitive Support',
    7: 'Low-Effort Slop & Hype',
    8: 'Other (Good)',
    9: 'Other (Bad)',
}

# 1. Define the DSPy Signature
class LocalLlamaPostClassifier(dspy.Signature):
    """
    Classify a Reddit post from r/localllama into exactly one of the following categories:

    1. Projects & Showcases: Legitimate, functioning side projects, UI tools, and applications built by users leveraging local LLMs (e.g., hardware integrations, persistent game engines, usable coding harnesses). This is the core "hacker" content of the community.
    2. Technical Deep Dives & Hardware: Inference benchmarking on real setups, detailed discussions on quantization (AutoRound, QAT), memory management strategies, system architecture, and specific/niche troubleshooting. High-effort knowledge sharing and technical problem-solving.
    3. Relevant News & Releases: Actual releases of open-weight models, major software updates (like llama.cpp merges), and significant industry news directly impacting the open-source AI ecosystem. High-signal updates that the community needs to stay informed.
    4. Meta & General Discussions: Opinion pieces on the state of the AI industry, debates about API pricing vs. local hosting, or discussions about subreddit quality. These require human or secondary review because a high-effort discussion is valuable, but a low-effort rant is not.
    5. High-Effort Technical Inquiries: Advanced or novel technical support questions where the user provides detailed context, shares previous troubleshooting steps they have already tried, or asks a highly relevant question capable of sparking good discussion. 
    6. Beginner & Repetitive Support: The daily barrage of "What model can I run on my RTX 3060?" or vague "What is the best model?" questions that demonstrate zero prior research. It clutters the feed with easily searchable information.
    7. Low-Effort Slop & Hype: Clickbait social media screenshots (e.g., CEO quotes), fake/unrealistic model benchmarks, AI-generated/vibecoded projects with no real usability, and news about completely closed-source models (like OpenAI's financials). This is the ultimate "junk drawer" for karma farming, shilling, and irrelevant hype that provides no technical value.
    8. Other (Good): A high-quality post that absolutely cannot fit into categories 1-5.
    9. Other (Bad): A low-quality post that absolutely cannot fit into categories 6-7.

    CRITICAL INSTRUCTION: You must strictly choose one of the 6 core categories if at all possible. ONLY use "Other (Good)" or "Other (Bad)" if the post absolutely cannot fit into the boundaries of categories 1-6.
    """
    post_content: str = dspy.InputField(desc="The text content of the Reddit post.")
    comment: str = dspy.InputField(desc="A subjective comment on the post. Use this as reference when classifying.")
    category: int = dspy.OutputField(desc="Output ONLY the exact category number from the list above.")

# 2. Define the DSPy Module with Chain of Thought
class PostClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # Wrap the signature in ChainOfThought to force the LLM to reason first
        self.classifier = dspy.ChainOfThought(LocalLlamaPostClassifier)

    def forward(self, post_content, comment):
        return self.classifier(post_content=post_content, comment=comment)

# 3. Sample Data & Train/Test Split using scikit-learn
# Replace this with your actual loaded data (e.g., from a Pandas DataFrame)
posts_data = []
posts_dir = Path("LocalLlama_Posts_with_Commentary/")
for post_path in posts_dir.iterdir():
    if not post_path.is_file():
        continue
    post_text = post_path.open(encoding='utf-8').read().strip()
    post_lines = post_text.split("\n")
    commentary = post_lines[0].strip()
    assert (not post_lines[1].strip())
    post_content = "\n".join(post_lines[2:])
    posts_data.append({
        'content': post_content.strip(),
        'comment': commentary,
    })

#posts = [
#    "Upgrading path for RTX Pro 4500 Pro for coding (Qwen3.6-27B)...",
#    "Anthropic is intentionally nerfing Fable when asked to develop other LLMs...",
#    "Qwen 3.6 35B-A3B @ Q4 or Gemma 4 12B @ Q8? Wondering how much...",
#    "Deep Neural Network that can turn any Image into a Playable Game!...",
#    "Vercel CEO: 'Almost shocked' by how good GLM-5.2 is at coding..."
#]

# Optional: If you already have manual labels for your dataset to evaluate against
labels = [
    "Technical Deep Dives & Hardware",
    "Low-Effort Slop & Hype",
    "Beginner & Repetitive Support",
    "Projects & Showcases",
    "Low-Effort Slop & Hype"
]

# Split the data (80% train, 20% test)
#X_train, X_test = train_test_split(posts, test_size=0.2, random_state=34)
#print(f"Train size: {len(X_train)}")
#print(f"Test size: {len(X_test)}")

# 4. Example Execution
def main():
    api_base = os.environ['OPENAI_API_BASE']
    api_key = os.environ['OPENAI_API_KEY']
    model_name = os.environ['OPENAI_MODEL']
    if not model_name.startswith("openai/"):
        model_name = "openai/" + model_name
    # Configure your LLM backend (e.g., OpenAI, local model via Ollama/vLLM, etc.)
    lm = dspy.LM(model_name,
                 api_base=api_base, api_key=api_key)
    dspy.configure(lm=lm)
    
    # Initialize the module
    clf = PostClassifier()
    
    posts_cls_data = []
    for post_idx, post_d in enumerate(tqdm(posts_data, desc="Classifying posts")):
        cls_result = clf(post_content=post_d['content'],
                         comment=post_d['comment'])
        category_num = cls_result.category
        category_name = category_num_to_name.get(category_num, 7)
        print(f"--- Post Content ---\n{post_d['content']}\n")
        print(f"--- LLM Reasoning ---\n{cls_result.reasoning}\n")
        print(f"--- Final Category ---\n{category_name} ({category_num})")
        posts_cls_data.append({
            **post_d,
            'classification': category_name,
            'classification_reasoning': cls_result.reasoning,
        })

    # Save results
    results_save_path = "classification_results.json"
    with open(results_save_path, 'w', encoding='utf-8') as f:
        json.dump(posts_cls_data, f, indent=2, ensure_ascii=True)
    print("Saved results to: {}".format(results_save_path))


if __name__ == "__main__":
    main()
