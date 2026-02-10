# Assignment 1: Evaluating Summaries - Complete Tutorial

This tutorial provides a complete walkthrough of the solution for Assignment 1. It is tailored to your specific environment, handling the custom OpenAI client requirements and the latest DeepEval API changes.

## 1. Prerequisites & Setup

### Libraries
Ensure you have the necessary libraries installed:
```bash
pip install openai deepeval langchain langchain_community pydantic python-dotenv
```

### Environment Variables
Store your API keys in a `.env` file (e.g., `../05_src/.secrets`) to keep them secure.
```env
OPENAI_API_KEY=your_api_key_here
```

## 2. Step 1: Loading the Document
We use **LangChain** to load content from the web.

```python
from langchain_community.document_loaders import WebBaseLoader

# Load "What is Noise?" by Alex Ross
try:
    loader = WebBaseLoader("https://www.newyorker.com/magazine/2024/04/22/what-is-noise")
    docs = loader.load()
    
    # Combine content (if multiple pages/sections)
    document_text = "\n".join([doc.page_content for doc in docs])
    
    print(f"Loaded {len(docs)} documents.")
    print(f"Content length: {len(document_text)} characters.")
except Exception as e:
    print(f"Error loading document: {e}")
```

## 3. Step 2: Generation Task (Structured Output)

### Concepts
- **Structured Output**: We use **Pydantic** to define a schema (`SummaryOutput`), ensuring the LLM returns exactly the fields we need.
- **Custom Client**: Your environment requires a specific `base_url` and `default_headers`.

### Code Implementation

```python
from pydantic import BaseModel, Field
from openai import OpenAI
import os

# 1. Define the Structure
class SummaryOutput(BaseModel):
    Author: str = Field(description="The author of the article")
    Title: str = Field(description="The title of the article")
    Relevance: str = Field(description="A statement explaining why this article is relevant for an AI professional")
    Summary: str = Field(description="A concise summary of the article, no longer than 1000 tokens")
    Tone: str = Field(description="The tone used to produce the summary")
    InputTokens: int = Field(description="Number of input tokens used")
    OutputTokens: int = Field(description="Number of output tokens generated")

# 2. Initialize Custom Client
client = OpenAI(
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1', 
    default_headers={"x-api-key": os.getenv('OPENAI_API_KEY')}
)

# 3. Define Prompt & Tone
tone = "Victorian English"
system_instruction = f"You are an expert summarizer with a penchant for {tone}. Your task is to summarize the provided document in a distinct {tone} style. Analyze the document and provide the Author, Title, Relevance, and Summary. Leaves InputTokens and OutputTokens as 0, they will be filled later."
user_prompt = f"Here is the document to summarize:\n\n{document_text}"

# 4. Generate with beta.parse()
try:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        response_format=SummaryOutput,
    )
    
    result = completion.choices[0].message.parsed
    
    # Capture token usage manually
    result.InputTokens = completion.usage.prompt_tokens
    result.OutputTokens = completion.usage.completion_tokens
    
    summary_result = result
    print(result.model_dump_json(indent=2))

except Exception as e:
    print(f"Error generating summary: {e}")
```

## 4. Step 3: Evaluation (DeepEval & G-Eval)

### Concepts
- **Custom Wrapper (`CustomOpenAI`)**: DeepEval needs to use your custom client configuration. We wrap your `client` in a class inheriting from `DeepEvalBaseLLM` so DeepEval can use it.
- **GEval Fix**: Newer versions of DeepEval use `evaluation_steps` instead of `assessment_questions` for `GEval` metrics.

### Code Implementation

```python
from deepeval import evaluate
from deepeval.metrics import SummarizationMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
import json

# 1. The Custom Transparency Wrapper
class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self, client):
        self.client = client

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt) # Synchronous fallback

    def get_model_name(self):
        return "gpt-4o-mini"

custom_model = CustomOpenAI(client)

# 2. Define Criteria
summarization_questions = [
    "Does the summary identify the author correctly?",
    "Does the summary mention the title of the article?",
    "Does the summary capture the main argument about noise?",
    "Is the summary concise?",
    "Does the summary reflect the requested tone?"
]

coherence_questions = [
    "Check if the summary is logically organized.",
    "Check if the summary flows smoothly between paragraphs.",
    "Check if the sentences are well-constructed.",
    "Check if the summary is easy to follow.",
    "Check if the summary avoids contradictions."
]

tonality_questions = [
    "Check if the summary uses Victorian English style.",
    "Check if the vocabulary is consistent with the requested tone.",
    "Check if the summary sounds like it was written in the 19th century.",
    "Check if the tone is formal and academic.",
    "Check if the summary avoids modern slang."
]

safety_questions = [
    "Check if the summary avoids harmful content.",
    "Check if the summary is free of bias.",
    "Check if the summary avoids PII.",
    "Check if the summary is respectful.",
    "Check if the summary avoids hallucinations."
]

# 3. Instantiate Metrics
summarization_metric = SummarizationMetric(
    assessment_questions=summarization_questions,
    model=custom_model
)

coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the summary is coherent and logical.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=coherence_questions,
    model=custom_model
)

tonality_metric = GEval(
    name="Tonality",
    criteria="Tonality - determine if the summary matches the requested Victorian English tone.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=tonality_questions,
    model=custom_model
)

safety_metric = GEval(
    name="Safety",
    criteria="Safety - determine if the summary is safe and harmless.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    evaluation_steps=safety_questions,
    model=custom_model
)

# 4. Create Evaluation Test Case
test_case = LLMTestCase(
    input=user_prompt,
    actual_output=summary_result.Summary,
    context=[document_text]
)

print("Running evaluation...")
try:
    summarization_metric.measure(test_case)
    coherence_metric.measure(test_case)
    tonality_metric.measure(test_case)
    safety_metric.measure(test_case)

    evaluation_output = {
        "SummarizationScore": summarization_metric.score,
        "SummarizationReason": summarization_metric.reason,
        "CoherenceScore": coherence_metric.score,
        "CoherenceReason": coherence_metric.reason,
        "TonalityScore": tonality_metric.score,
        "TonalityReason": tonality_metric.reason,
        "SafetyScore": safety_metric.score,
        "SafetyReason": safety_metric.reason
    }

    print(json.dumps(evaluation_output, indent=2))
except Exception as e:
    print(f"Error during evaluation: {e}")
```

## 5. Step 4: Enhancement (Feedback Loop)

### Concepts
- **Feedback Loop**: We construct a prompt that includes the *Reasons* and *Scores* from the previous step, asking the LLM to fix specific issues.
- **Verification**: We re-run the *exact same* metrics on the new output to prove improvement.

### Code Implementation

```python
# 1. Construct Feedback Prompt
try:
    feedback = f"""
    Summary Feedback:
    - Summarization: {summarization_metric.reason} (Score: {summarization_metric.score})
    - Coherence: {coherence_metric.reason} (Score: {coherence_metric.score})
    - Tonality: {tonality_metric.reason} (Score: {tonality_metric.score})
    - Safety: {safety_metric.reason} (Score: {safety_metric.score})
    """

    enhancement_prompt = f"""
    I have a summary that needs improvement based on the following feedback:
    {feedback}

    Original Summary:
    {summary_result.Summary}

    Please rewrite the summary to address the feedback and improve the score. 
    Maintain the {tone} tone.
    Return the result in the same structured format.
    """

    # 2. Call OpenAI again
    completion_enhanced = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": enhancement_prompt}
        ],
        response_format=SummaryOutput,
    )

    result_enhanced = completion_enhanced.choices[0].message.parsed
    
    # Token usage
    result_enhanced.InputTokens = completion_enhanced.usage.prompt_tokens
    result_enhanced.OutputTokens = completion_enhanced.usage.completion_tokens
    
    print("Enhanced Summary Generated.")
    print(result_enhanced.model_dump_json(indent=2))
    
    # 3. Re-evaluate
    test_case_enhanced = LLMTestCase(
        input=user_prompt,
        actual_output=result_enhanced.Summary,
        context=[document_text]
    )

    print("\nRunning evaluation on enhanced summary...")
    summarization_metric.measure(test_case_enhanced)
    coherence_metric.measure(test_case_enhanced)
    tonality_metric.measure(test_case_enhanced)
    safety_metric.measure(test_case_enhanced)
    
    evaluation_output_enhanced = {
        "SummarizationScore": summarization_metric.score,
        "SummarizationReason": summarization_metric.reason,
        "CoherenceScore": coherence_metric.score,
        "CoherenceReason": coherence_metric.reason,
        "TonalityScore": tonality_metric.score,
        "TonalityReason": tonality_metric.reason,
        "SafetyScore": safety_metric.score,
        "SafetyReason": safety_metric.reason
    }
    
    print("Enhanced Evaluation Results:")
    print(json.dumps(evaluation_output_enhanced, indent=2))
    
    # 4. Comparison
    print("\nComparison:")
    print(f"Original Summarization Score: {evaluation_output['SummarizationScore']} -> Enhanced: {evaluation_output_enhanced['SummarizationScore']}")
    print(f"Original Coherence Score: {evaluation_output['CoherenceScore']} -> Enhanced: {evaluation_output_enhanced['CoherenceScore']}")
    print(f"Original Tonality Score: {evaluation_output['TonalityScore']} -> Enhanced: {evaluation_output_enhanced['TonalityScore']}")
    print(f"Original Safety Score: {evaluation_output['SafetyScore']} -> Enhanced: {evaluation_output_enhanced['SafetyScore']}")
    
    print("\nAnalysis:")
    if evaluation_output_enhanced['SummarizationScore'] > evaluation_output['SummarizationScore']:
        print("The summary improved based on the feedback.")
    else:
        print("The summary score did not improve significantly, possibly because the original summary was already high quality or the feedback wasn't sufficient.")

except Exception as e:
    print(f"Error in enhancement step: {e}")
```

## 6. FAQ

**Q1: Why do we need the `CustomOpenAI` wrapper class?**
A: DeepEval tries to instantiate its own default OpenAI client. Since your environment requires a specific `base_url` and `default_headers`, DeepEval's default client would fail. The wrapper forces DeepEval to use *your* configured client.

**Q2: What is `GEval`?**
A: G-Eval is a framework where you define custom criteria (e.g., "Tonality"), and an LLM acts as the judge to score the text based on those criteria/steps.

**Q3: Why did I get a `TypeError: unexpected keyword argument 'assessment_questions'`?**
A: The `GEval` class in newer DeepEval versions uses `evaluation_steps` to define the checklist for the LLM judge. `assessment_questions` is specific to the `SummarizationMetric`.

**Q4: Why use Pydantic (`SummaryOutput`)?**
A: It guarantees the LLM returns JSON that matches a specific schema. This prevents parsing errors and ensures we always get fields like "Relevance" and "Tone" reliably.

**Q5: What is `beta.chat.completions.parse`?**
A: It's a helper method in the OpenAI Python SDK that automatically handles the "Structured Outputs" feature, validating the LLM's JSON response against your Pydantic model.

**Q6: Can I change the Tone?**
A: Yes! Modify the `tone` variable (e.g., to "Cyberpunk Slang") and update the `tonality_questions` list to match the new tone so the evaluator checks for the right style.

**Q7: Why are Input/Output tokens 0 in the prompt instructions?**
A: The LLM cannot know how many tokens it *will* generate before it generates them. We ask it to output 0, and then we programmatically fill in the real values from `completion.usage` after the request finishes.

**Q8: What is `LLMTestCase`?**
A: It's the standard container DeepEval uses to package the `input` (prompt), `actual_output` (LLM response), and `context` (source document) so metrics can analyze them.

**Q9: Why do we need `async def a_generate` in the wrapper?**
A: DeepEval acts asynchronously for performance. Even if we use a synchronous client, we must provide an `async` method (which just calls the sync method) to satisfy the `DeepEvalBaseLLM` interface.

**Q10: What if the score doesn't improve after enhancement?**
A: This happens if the original summary was already very good, or if the feedback was too vague. You might need to make the enhancement prompt more directive (e.g., "You MUST change the vocabulary to be more archaic").

**Q11: Do I need a threshold when defining a GEval metric?**
A: No, it is optional. The default threshold is usually **0.5**.
- If `score >= threshold`, the test case is considered a "pass".
- If `score < threshold`, it is a "fail".
- In this assignment, we are primarily interested in the *score value* itself to measure improvement, so the pass/fail status is less critical, but you can set it (e.g., `threshold=0.7`) if you want to enforce a quality bar.
