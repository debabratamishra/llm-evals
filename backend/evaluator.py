import time
import json
import difflib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# LiteLLM as the unified LLM gateway
try:
    import litellm
    from litellm import completion
    litellm.drop_params = True  # silently ignore unsupported params per provider
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class EvaluationRunner:
    """
    Unified evaluation runner backed entirely by LiteLLM.

    Provider routing:
      - "mock"         → deterministic sandbox, no keys needed
      - "ollama"       → Ollama local via LiteLLM   (no key, configurable base_url)
      - "ollama_cloud" → Ollama cloud via LiteLLM   (env/param: OLLAMA_API_KEY + custom base_url)
      - "openrouter"   → OpenRouter via LiteLLM     (env/param: OPENROUTER_API_KEY, free model choice)
    """

    def __init__(
        self,
        ollama_api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        # Ollama (local defaults to http://localhost:11434, cloud needs key + custom URL)
        self.ollama_api_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY")
        self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # OpenRouter
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

    # ------------------------------------------------------------------
    # Internal: unified LiteLLM call
    # ------------------------------------------------------------------

    def _call_litellm(
        self,
        litellm_model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Low-level wrapper around litellm.completion()."""
        if not HAS_LITELLM:
            raise RuntimeError("litellm is not installed. Run: pip install litellm")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": litellm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        start_time = time.time()
        try:
            response = completion(**kwargs)
            latency = time.time() - start_time

            text = response.choices[0].message.content or ""
            usage = response.usage or {}
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

            # Estimate when the provider doesn't return usage
            if input_tokens == 0:
                input_tokens = int(len(prompt.split()) * 1.3)
            if output_tokens == 0:
                output_tokens = int(len(text.split()) * 1.3)

            return {
                "text": text,
                "latency": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }
        except Exception as e:
            return {
                "text": f"Error calling {litellm_model}: {str(e)}",
                "latency": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Provider-specific callers (thin wrappers over _call_litellm)
    # ------------------------------------------------------------------

    def _call_ollama(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Calls a local Ollama instance via LiteLLM.
        Expects ollama_base_url to point at http://localhost:11434 (default).
        """
        # LiteLLM requires "ollama/<model>" prefix for local
        litellm_model = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"
        return self._call_litellm(
            litellm_model=litellm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            api_base=self.ollama_base_url,
        )

    def _call_ollama_cloud(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Calls a cloud-hosted Ollama-compatible endpoint via LiteLLM.
        Requires ollama_api_key and a custom ollama_base_url.
        """
        if not self.ollama_api_key:
            raise ValueError("Ollama cloud API key is not configured.")
        if not self.ollama_base_url or self.ollama_base_url == "http://localhost:11434":
            raise ValueError(
                "A custom Ollama cloud base URL is required. "
                "Local default URL is not valid for cloud provider."
            )
        # Use openai-compatible prefix so LiteLLM routes via the custom base
        litellm_model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
        return self._call_litellm(
            litellm_model=litellm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            api_key=self.ollama_api_key,
            api_base=self.ollama_base_url,
        )

    def _call_openrouter(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Calls OpenRouter via LiteLLM."""
        if not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is not configured.")
        # LiteLLM expects "openrouter/<model>" prefix
        litellm_model = (
            model_name if model_name.startswith("openrouter/") else f"openrouter/{model_name}"
        )
        return self._call_litellm(
            litellm_model=litellm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            api_key=self.openrouter_api_key,
            extra_headers={
                "HTTP-Referer": "https://github.com/llm-evals",
                "X-Title": "LLM Evals Framework",
            },
        )

    def _call_mock(self, model_name: str, question: str, ideal_answer: str) -> Dict[str, Any]:
        """Simulates a model response for development/sandbox use — no API keys required."""
        start_time = time.time()
        time.sleep(0.5)
        latency = time.time() - start_time + 0.5

        if "llama" in model_name.lower():
            prefix = "[Llama Mock] "
        elif "phi" in model_name.lower():
            prefix = "[Phi Mock] "
        elif "mistral" in model_name.lower():
            prefix = "[Mistral Mock] "
        else:
            prefix = "[Mock Model Response] "

        if len(question) % 3 == 0:
            mock_answer = (
                f"{prefix}Based on evaluation guidelines, "
                f"{ideal_answer.replace('The suspected', 'our primary suspected')} "
                "This represents the ideal expected response."
            )
        elif len(question) % 3 == 1:
            mock_answer = f"{prefix}{ideal_answer}"
        else:
            words = ideal_answer.split()
            halflen = len(words) // 2
            mock_answer = (
                f"{prefix}Based on initial context, we suspect a standard outcome. "
                f"{' '.join(words[:halflen])}... "
                "(Note: further evaluation is required to confirm details)."
            )

        return {
            "text": mock_answer,
            "latency": latency,
            "input_tokens": int(len(question.split()) * 1.3),
            "output_tokens": int(len(mock_answer.split()) * 1.3),
            "error": None,
        }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _get_exact_match(self, candidate: str, reference: str) -> int:
        """Binary exact match (1 or 0) after alphanumeric normalization."""
        def normalize(t):
            return "".join(c.lower() for c in t if c.isalnum())
        return 1 if normalize(candidate) == normalize(reference) else 0

    def _get_similarity_score(self, candidate: str, reference: str) -> float:
        """Sequence-based similarity score in [0, 1]."""
        return round(difflib.SequenceMatcher(None, candidate, reference).ratio(), 3)

    # ------------------------------------------------------------------
    # LLM-as-a-Judge
    # ------------------------------------------------------------------

    def _run_llm_as_judge(
        self, question: str, ideal_answer: str, model_answer: str
    ) -> Dict[str, Any]:
        """
        Rates the candidate answer on correctness, completeness, and clarity.
        Tries available providers in order: Gemini → OpenAI → OpenRouter → Ollama → heuristic.
        """
        judge_prompt = f"""You are an expert AI evaluator grading the output of a language model.
Given a user query, a golden reference answer, and the candidate model's answer, rate the candidate answer on three dimensions:
1. Correctness (1-5): Factual correctness. Does it align with the facts in the golden answer? (1 = completely wrong, 5 = perfectly correct)
2. Completeness (1-5): How comprehensive is the response? Does it cover all points mentioned in the golden answer? (1 = covers nothing, 5 = covers all points)
3. Clarity (1-5): Is the answer clear, coherent, and professional? (1 = unreadable/gibberish, 5 = exceptionally clear)

Provide your output ONLY as a valid JSON object matching this schema:
{{
  "correctness": integer,
  "completeness": integer,
  "clarity": integer,
  "reason": "brief explanation of the grading"
}}

---
User Query: {question}
Golden Reference Answer: {ideal_answer}
Candidate Model Answer: {model_answer}
---
JSON Response:"""

        def _parse_judge(text: str) -> Optional[Dict[str, Any]]:
            try:
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                parsed = json.loads(text.strip())
                return {
                    "correctness": int(parsed.get("correctness", 4)),
                    "completeness": int(parsed.get("completeness", 4)),
                    "clarity": int(parsed.get("clarity", 4)),
                    "reason": parsed.get("reason", "Grader successfully evaluated response."),
                }
            except Exception:
                return None

        # 1. Try OpenRouter
        if self.openrouter_api_key:
            try:
                res = _parse_judge(
                    self._call_openrouter(
                        "meta-llama/llama-3.1-8b-instruct:free", judge_prompt, temperature=0.1
                    )["text"]
                )
                if res:
                    return res
            except Exception as e:
                print(f"OpenRouter Judge failed: {e}")

        # 2. Try local Ollama
        try:
            res = _parse_judge(
                self._call_ollama("llama3.2", judge_prompt, temperature=0.1)["text"]
            )
            if res:
                return res
        except Exception as e:
            print(f"Ollama Judge failed: {e}")

        # 5. Heuristic fallback
        sim = self._get_similarity_score(model_answer, ideal_answer)
        exact = self._get_exact_match(model_answer, ideal_answer)

        if exact == 1:
            return {"correctness": 5, "completeness": 5, "clarity": 5,
                    "reason": "Exact match detected. Fully correct and complete."}
        if sim > 0.8:
            return {"correctness": 5, "completeness": 5, "clarity": 5,
                    "reason": "Heuristic Judge: Extremely high overlap with golden answer."}
        if sim > 0.6:
            return {"correctness": 4, "completeness": 4, "clarity": 5,
                    "reason": "Heuristic Judge: High semantic similarity. Captures primary concepts."}
        if sim > 0.4:
            return {"correctness": 3, "completeness": 3, "clarity": 4,
                    "reason": "Heuristic Judge: Moderate similarity. Some key details missing."}
        return {"correctness": 2, "completeness": 2, "clarity": 3,
                "reason": "Heuristic Judge: Low similarity. Response diverges from reference answer."}

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _calculate_costs(
        self, provider: str, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Approximate cost in USD based on known pricing (per 1M tokens)."""
        p = provider.lower()
        m = model_name.lower()

        pricing: Dict[str, Any] = {
            # Ollama (local) is free; cloud instances vary — use $0 as default
            "ollama": {"default": {"input": 0.0, "output": 0.0}},
            "ollama_cloud": {"default": {"input": 0.0, "output": 0.0}},
            # OpenRouter pricing varies widely; use conservative estimates per model
            "openrouter": {
                "default": {"input": 0.50, "output": 1.50},
                "llama-3.1-8b": {"input": 0.05, "output": 0.05},
                "mistral-7b": {"input": 0.06, "output": 0.06},
                "claude-3-haiku": {"input": 0.25, "output": 1.25},
                "gpt-4o": {"input": 5.00, "output": 15.00},
                "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            },
            "mock": {"default": {"input": 0.0, "output": 0.0}},
        }

        prov_rates = pricing.get(p, pricing["mock"])
        rates = prov_rates.get("default", {"input": 0.0, "output": 0.0})
        for k in prov_rates:
            if k != "default" and k in m:
                rates = prov_rates[k]
                break

        return round(
            (input_tokens / 1_000_000) * rates["input"]
            + (output_tokens / 1_000_000) * rates["output"],
            6,
        )

    # ------------------------------------------------------------------
    # Main evaluation pipeline
    # ------------------------------------------------------------------

    def run_evaluation(self, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the full evaluation pipeline over a golden Q&A dataset."""
        model_provider = config.get("model_provider", "mock")
        model_name = config.get("model_name", "mock-model")
        temperature = config.get("temperature", 0.2)
        system_prompt = config.get("system_prompt", "")

        cases = dataset.get("cases", [])
        results = []

        total_latency = 0.0
        total_cost = 0.0
        total_exact = 0
        sum_correctness = 0.0
        sum_completeness = 0.0
        sum_clarity = 0.0
        sum_similarity = 0.0
        total_cases = len(cases)

        for i, case in enumerate(cases):
            question = case["question"]
            ideal_answer = case["ideal_answer"]
            case_id = case.get("id", str(i))

            # -- Step 1: call the target model --
            if model_provider == "ollama":
                model_res = self._call_ollama(model_name, question, system_prompt, temperature)
            elif model_provider == "ollama_cloud":
                model_res = self._call_ollama_cloud(model_name, question, system_prompt, temperature)
            elif model_provider == "openrouter":
                model_res = self._call_openrouter(model_name, question, system_prompt, temperature)
            else:
                model_res = self._call_mock(model_name, question, ideal_answer)

            model_answer = model_res["text"]
            latency = model_res["latency"]
            input_tokens = model_res["input_tokens"]
            output_tokens = model_res["output_tokens"]

            # -- Step 2: standard metrics --
            exact_match = self._get_exact_match(model_answer, ideal_answer)
            similarity = self._get_similarity_score(model_answer, ideal_answer)

            # -- Step 3: LLM-as-a-judge --
            judge_res = self._run_llm_as_judge(question, ideal_answer, model_answer)

            # -- Step 4: cost --
            cost = self._calculate_costs(model_provider, model_name, input_tokens, output_tokens)

            total_latency += latency
            total_cost += cost
            total_exact += exact_match
            sum_correctness += judge_res["correctness"]
            sum_completeness += judge_res["completeness"]
            sum_clarity += judge_res["clarity"]
            sum_similarity += similarity

            results.append({
                "case_id": case_id,
                "question": question,
                "ideal_answer": ideal_answer,
                "model_answer": model_answer,
                "metrics": {
                    "exact_match": exact_match,
                    "similarity": similarity,
                    "llm_correctness": judge_res["correctness"],
                    "llm_completeness": judge_res["completeness"],
                    "llm_clarity": judge_res["clarity"],
                    "latency": round(latency, 2),
                    "cost": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reason": judge_res["reason"],
                },
            })

        n = total_cases or 1  # avoid division by zero
        return {
            "name": config.get("run_name", f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "model_provider": model_provider,
            "model_name": model_name,
            "parameters": {
                "temperature": temperature,
                "system_prompt": system_prompt,
            },
            "created_at": datetime.utcnow().isoformat(),
            "metrics": {
                "avg_accuracy": round(total_exact / n, 3),
                "avg_similarity": round(sum_similarity / n, 2),
                "avg_correctness": round(sum_correctness / n, 2),
                "avg_completeness": round(sum_completeness / n, 2),
                "avg_clarity": round(sum_clarity / n, 2),
                "avg_latency": round(total_latency / n, 2),
                "total_cost": round(total_cost, 6),
                "total_cases": total_cases,
            },
            "results": results,
        }
