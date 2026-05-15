# I Replaced $800/mo in API Costs with a Local Llama 4 Setup for E-Commerce

My team runs an e-commerce operation that pushes around 80,000 product descriptions through LLMs every month. We were spending $800+ on GPT-4o API calls. Last month we moved the bulk generation pipeline to Llama 4 Maverick running locally via Ollama. Monthly cost dropped to about $40 in electricity.

Here's the full setup, what worked, what didn't, and where we still use cloud APIs.

## Why bother running locally

Three reasons pushed us off the API-only approach:

**Cost at scale.** 80K descriptions at ~500 tokens each, GPT-4o was billing us $600-800/month. Fine for a startup burning cash, not great when you're trying to run profitably.

**Data privacy.** We process competitor pricing data and customer purchase history for segmentation. Sending that to a third-party API means it leaves your infrastructure. With GDPR customers in our mix, local processing just removes an entire category of compliance headaches.

**Rate limits and latency.** During product launch weeks we'd hit rate limits and queue up requests. A local model doesn't throttle you — it just runs as fast as your GPU allows.

## Hardware and what to expect

We tested on three setups:

| Machine | VRAM | Speed | Notes |
|---------|------|-------|-------|
| Mac M3 Max 64GB | Unified | ~18 tok/s | Fine for dev/testing, too slow for batch |
| RTX 4090 24GB | 24GB | ~35 tok/s | Our production choice. Handles 800-1200 descriptions/hr |
| 2x RTX 4090 | 48GB | ~55 tok/s | Overkill for our volume, but nice for parallel jobs |

If your VRAM is too small, Ollama silently falls back to CPU. You'll see 3-5 tokens/second and wonder what went wrong. Check `ollama ps` to verify the model loaded onto GPU.

## Setup (takes about 10 minutes)

Install Ollama:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the Hermes fine-tune of Maverick (this is the version you want — base Maverick has flaky JSON output):

```bash
ollama pull hermes3:maverick
# ~25GB download, grab a coffee
```

Start the server and test:

```bash
ollama serve &

curl http://localhost:11434/api/generate -d '{
  "model": "hermes3:maverick",
  "prompt": "Generate a product title for: wireless bluetooth earbuds, IPX7 waterproof, 30hr battery, noise cancelling. Output JSON: {\"title\": \"...\", \"keywords\": [...]}",
  "stream": false
}'
```

## Why Hermes and not base Maverick

We wasted two days on base Maverick before switching. The difference:

- **JSON output:** Base Maverick returns valid JSON about 88% of the time. Hermes hits 97%+. When you're generating 80K items, that 9% gap means thousands of failed parses and retries.
- **Function calling:** We use tool calls to pull inventory data mid-generation. Base model: 78% accuracy. Hermes: 93%.
- **System prompt adherence:** Tell base Maverick "always respond in German" and it drifts back to English after ~20 turns. Hermes stays consistent.

## The actual pipeline code

Our production script is a Python worker that reads from a job queue and writes to a database. Here's the core of it:

```python
import httpx
import json

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

def generate_description(product: dict, lang: str = "en") -> dict:
    prompt = f"""Write a product description for an e-commerce listing.
Product: {json.dumps(product)}
Language: {lang}
Output JSON: {{"title": "...", "description": "...", "bullet_points": [...]}}
Only output the JSON object, nothing else."""

    resp = httpx.post(OLLAMA_URL, json={
        "model": "hermes3:maverick",
        "messages": [
            {"role": "system", "content": "You are a product copywriter. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }, timeout=60)

    text = resp.json()["choices"][0]["message"]["content"]
    # Strip markdown fences if the model wraps output
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)

# Example usage
product = {
    "name": "Wireless Earbuds Pro",
    "material": "ABS plastic, silicone tips",
    "features": ["IPX7 waterproof", "30hr battery", "ANC"],
    "price_range": "$25-35"
}

result = generate_description(product, lang="de")
print(json.dumps(result, indent=2, ensure_ascii=False))
```

Ollama's API is OpenAI-compatible, so if you have existing code that calls `openai.ChatCompletion`, change the base URL and model name. That's it.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="hermes3:maverick",
    messages=[{"role": "user", "content": "your prompt here"}]
)
```

## Where local still loses

We didn't move everything off cloud APIs. Three tasks still run through Claude or GPT-4o:

1. **Brand voice copy** — when the output needs to sound like a specific brand, cloud models are noticeably better. Maverick writes competent descriptions but they read a bit flat compared to Claude's output.
2. **Anything under 10K requests/month** — the break-even point is somewhere around 50K monthly requests. Below that, GPT-4o-mini at $150/mo beats the hassle of maintaining local hardware.
3. **One-off creative tasks** — ad headlines, email subject lines, anything where you want to iterate on quality. Cloud models with their bigger parameter counts just produce more varied and interesting options.

## Monthly cost comparison (our actual numbers)

| | Before (API only) | After (hybrid) |
|---|---|---|
| Bulk descriptions (80K) | $620 (GPT-4o) | $40 electricity |
| Creative copy (5K) | $180 (Claude Sonnet) | $180 (Claude Sonnet) |
| Ad headlines (2K) | $30 (GPT-4o-mini) | $30 (GPT-4o-mini) |
| **Total** | **$830/mo** | **$250/mo** |

Hardware was a one-time $1,800 for the RTX 4090 rig. Paid for itself in three months.

## Resources

If you're building something similar, I put together a few things that might help:

- [awesome-ai-ecommerce-tools](https://github.com/xinpengdr/awesome-ai-ecommerce-tools) — curated list of 42+ AI tools for e-commerce, including local deployment options
- [Detailed Llama 4 vs Claude vs GPT cost breakdown](https://www.vozai.net/en/tool-comparisons/llama-4-maverick-ecommerce-ai-local/) — hardware configs, benchmark numbers, and use case recommendations
- [Ollama docs](https://ollama.com/) — the official setup and API reference
- [Hermes on HuggingFace](https://huggingface.co/NousResearch) — model weights and fine-tune details

---

The local LLM space moves fast. Six months ago, running a model this capable on a single consumer GPU wasn't realistic. Now it costs less than a Netflix subscription to process volumes that used to run up four-figure API bills. If you're hitting scale where API costs hurt, it's worth an afternoon to test.
