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
      - "nvidia_nim"   → Nvidia NIM via LiteLLM     (env/param: NVIDIA_NIM_API_KEY + optional base_url)
      - "openrouter"   → OpenRouter via LiteLLM     (env/param: OPENROUTER_API_KEY, free model choice)
    """

    def __init__(
        self,
        nvidia_nim_api_key: Optional[str] = None,
        nvidia_nim_base_url: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        # Nvidia NIM
        self.nvidia_nim_api_key = (
            nvidia_nim_api_key 
            or os.environ.get("NVIDIA_NIM_API_KEY") 
            or os.environ.get("NVIDIA_API_KEY")
        )
        self.nvidia_nim_base_url = (
            nvidia_nim_base_url 
            or os.environ.get("NVIDIA_NIM_API_BASE") 
            or os.environ.get("NVIDIA_API_BASE") 
            or "https://integrate.api.nvidia.com/v1"
        )
        # OpenRouter
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

    # ------------------------------------------------------------------
    # Internal: unified LiteLLM call
    # ------------------------------------------------------------------

    def _call_litellm(
        self,
        litellm_model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Low-level wrapper around litellm.completion()."""
        if not HAS_LITELLM:
            raise RuntimeError("litellm is not installed. Run: pip install litellm")

        kwargs: Dict[str, Any] = {
            "model": litellm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
            
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
                prompt_text = " ".join([m.get("content", "") for m in messages])
                input_tokens = int(len(prompt_text.split()) * 1.3)
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

    def _call_nvidia_nim(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calls Nvidia NIM via LiteLLM.
        Requires nvidia_nim_api_key and optional nvidia_nim_base_url.
        """
        # LiteLLM requires "nvidia_nim/<model>" prefix
        litellm_model = (
            model_name if model_name.startswith("nvidia_nim/") else f"nvidia_nim/{model_name}"
        )
        return self._call_litellm(
            litellm_model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            api_key=self.nvidia_nim_api_key,
            api_base=self.nvidia_nim_base_url,
        )

    def _call_openrouter(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
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
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
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
        Tries available providers in order: OpenRouter → Nvidia NIM → rubric heuristics.
        """
        judge_prompt = f"""You are an expert AI evaluator grading the output of a language model.
Given a user query, a golden reference answer, and the candidate model's answer, rate the candidate answer on three dimensions:
1. Correctness (1-5): Factual correctness compared to the golden answer.
2. Completeness (1-5): Comprehensiveness of content coverage compared to the golden answer.
3. Clarity (1-5): Formatting, coherence, and professional layout.

Strictly adhere to the following scoring rubrics for each score level:

### CORRECTNESS RUBRIC:
- 5 (Perfect): No factual errors, contradictions, or misleading claims compared to the golden answer.
- 4 (Minor Errors): The response is correct overall but contains minor inaccuracies, minor over-generalizations, or trivial omissions that do not compromise correctness.
- 3 (Moderate Errors): Contains some correct elements but also significant factual errors or claims that contradict the golden answer.
- 2 (Major Errors): Mostly incorrect. Only minor factual points align with the golden answer.
- 1 (Completely Incorrect): The response is entirely wrong, contains severe hallucinations, or is completely irrelevant to the question.

### COMPLETENESS RUBRIC:
- 5 (Perfect): Covers all key ideas, constraints, examples, and details specified in the golden answer.
- 4 (Minor Gaps): Covers all primary points but misses one or two minor/secondary details.
- 3 (Moderate Gaps): Covers about half of the major details in the golden answer; significant sections of the expected content are omitted.
- 2 (Major Gaps): Omit almost all necessary content; only a single correct detail is mentioned.
- 1 (Completely Incomplete): Fails to address any core parts of the reference answer.

### CLARITY RUBRIC:
- 5 (Perfect): Exceptionally clear, well-structured (e.g., bullet points/code blocks when relevant), highly professional tone, and excellent flow.
- 4 (Good): Easy to read and understand, with minor grammatical, flow, or formatting imperfections.
- 3 (Fair): Mostly readable but suffering from minor flow issues, repetitive structure, or slightly disorganized formatting.
- 2 (Poor): Disorganized, hard to read, or uses confusing sentence structures.
- 1 (Unreadable): Complete gibberish, unstructured text wall, or chaotic phrasing.

Provide your output ONLY as a valid JSON object matching this schema:
{{
  "correctness": integer,
  "completeness": integer,
  "clarity": integer,
  "reason": "detailed explanation of why this score was given, citing specific rubric matches"
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
                        "meta-llama/llama-3.2-3b-instruct", 
                        [{"role": "user", "content": judge_prompt}], 
                        temperature=0.1
                    )["text"]
                )
                if res:
                    return res
            except Exception as e:
                print(f"OpenRouter Judge failed: {e}")

        # 2. Try Nvidia NIM
        if self.nvidia_nim_api_key:
            try:
                res = _parse_judge(
                    self._call_nvidia_nim(
                        "meta/llama-3.2-3b-instruct", 
                        [{"role": "user", "content": judge_prompt}], 
                        temperature=0.1
                    )["text"]
                )
                if res:
                    return res
            except Exception as e:
                print(f"Nvidia NIM Judge failed: {e}")

        # 5. Heuristic fallback
        sim = self._get_similarity_score(model_answer, ideal_answer)
        exact = self._get_exact_match(model_answer, ideal_answer)

        if exact == 1:
            return {"correctness": 5, "completeness": 5, "clarity": 5,
                    "reason": "Heuristic Judge (Rubric Match: 5/5/5): Exact match detected. Fully correct, complete, and clear."}
        if sim > 0.8:
            return {"correctness": 5, "completeness": 5, "clarity": 5,
                    "reason": "Heuristic Judge (Rubric Match: 5/5/5): Extremely high overlap with golden answer."}
        if sim > 0.6:
            return {"correctness": 4, "completeness": 4, "clarity": 5,
                    "reason": "Heuristic Judge (Rubric Match: 4/4/5): High semantic similarity. Captures primary concepts with minor gaps."}
        if sim > 0.4:
            return {"correctness": 3, "completeness": 3, "clarity": 4,
                    "reason": "Heuristic Judge (Rubric Match: 3/3/4): Moderate similarity. Significant details from reference answer are missing."}
        return {"correctness": 2, "completeness": 2, "clarity": 3,
                "reason": "Heuristic Judge (Rubric Match: 2/2/3): Low similarity. Response diverges significantly from golden answer."}

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
            # Nvidia NIM cost estimation (default to 0.0)
            "nvidia_nim": {"default": {"input": 0.0, "output": 0.0}},
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
        """Runs the full evaluation pipeline over a golden dataset."""
        model_provider = config.get("model_provider", "mock")
        model_name = config.get("model_name", "mock-model")
        temperature = config.get("temperature", 0.2)
        system_prompt = config.get("system_prompt", "")
        max_tokens = config.get("max_tokens")
        top_p = config.get("top_p")
        frequency_penalty = config.get("frequency_penalty")
        presence_penalty = config.get("presence_penalty")
        multi_turn_history_mode = config.get("multi_turn_history_mode", "model_response")

        cases = dataset.get("cases", [])
        results = []

        total_latency = 0.0
        total_cost = 0.0
        total_exact = 0.0
        sum_correctness = 0.0
        sum_completeness = 0.0
        sum_clarity = 0.0
        sum_similarity = 0.0
        total_cases = len(cases)

        for i, case in enumerate(cases):
            turns = case.get("turns")
            is_case_multi_turn = turns is not None and len(turns) > 0

            if is_case_multi_turn:
                case_results = []
                actual_responses_history = []
                
                case_latency = 0.0
                case_cost = 0.0
                case_exact = 0
                case_similarity = 0.0
                case_correctness = 0.0
                case_completeness = 0.0
                case_clarity = 0.0
                
                for t_idx, turn in enumerate(turns):
                    user_msg = turn["user_message"]
                    ideal_resp = turn["ideal_response"]
                    
                    # Construct messages history up to this turn
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    for prev_t in range(t_idx):
                        messages.append({"role": "user", "content": turns[prev_t]["user_message"]})
                        
                        # Decide what to append based on the history mode selection
                        history_resp = (
                            actual_responses_history[prev_t]
                            if multi_turn_history_mode == "model_response"
                            else turns[prev_t]["ideal_response"]
                        )
                        messages.append({"role": "assistant", "content": history_resp})
                    messages.append({"role": "user", "content": user_msg})
                    
                    # Call target model
                    if model_provider == "nvidia_nim":
                        model_res = self._call_nvidia_nim(
                            model_name, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty
                        )
                    elif model_provider == "openrouter":
                        model_res = self._call_openrouter(
                            model_name, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty
                        )
                    else:
                        model_res = self._call_mock(model_name, user_msg, ideal_resp)
                        
                    model_answer = model_res["text"]
                    latency = model_res["latency"]
                    input_tokens = model_res["input_tokens"]
                    output_tokens = model_res["output_tokens"]
                    
                    actual_responses_history.append(model_answer)
                    
                    # Metrics
                    exact_match = self._get_exact_match(model_answer, ideal_resp)
                    similarity = self._get_similarity_score(model_answer, ideal_resp)
                    judge_res = self._run_llm_as_judge(user_msg, ideal_resp, model_answer)
                    cost = self._calculate_costs(model_provider, model_name, input_tokens, output_tokens)
                    
                    # Accumulate case totals
                    case_latency += latency
                    case_cost += cost
                    case_exact += exact_match
                    case_similarity += similarity
                    case_correctness += judge_res["correctness"]
                    case_completeness += judge_res["completeness"]
                    case_clarity += judge_res["clarity"]
                    
                    case_results.append({
                        "turn_index": t_idx,
                        "user_message": user_msg,
                        "ideal_response": ideal_resp,
                        "model_response": model_answer,
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
                        }
                    })
                
                # Turn averages for the case
                num_turns = len(turns)
                case_metrics = {
                    "exact_match": round(case_exact / num_turns, 2),
                    "similarity": round(case_similarity / num_turns, 2),
                    "llm_correctness": round(case_correctness / num_turns, 2),
                    "llm_completeness": round(case_completeness / num_turns, 2),
                    "llm_clarity": round(case_clarity / num_turns, 2),
                    "latency": round(case_latency, 2),
                    "cost": round(case_cost, 6),
                    "input_tokens": sum(t["metrics"]["input_tokens"] for t in case_results),
                    "output_tokens": sum(t["metrics"]["output_tokens"] for t in case_results),
                    "reason": f"Aggregated scores over {num_turns} conversational turns."
                }
                
                results.append({
                    "case_id": case.get("id", f"case-{i}"),
                    "is_multi_turn": True,
                    "turns": case_results,
                    # Fallback single-turn fields for basic displays
                    "question": turns[0]["user_message"],
                    "ideal_answer": turns[0]["ideal_response"],
                    "model_answer": case_results[0]["model_response"],
                    "metrics": case_metrics
                })
                
                # Accumulate overall run totals
                total_latency += case_latency
                total_cost += case_cost
                total_exact += case_exact / num_turns
                sum_correctness += case_correctness / num_turns
                sum_completeness += case_completeness / num_turns
                sum_clarity += case_clarity / num_turns
                sum_similarity += case_similarity / num_turns

            else:
                # SINGLE-TURN
                question = case["question"]
                ideal_answer = case["ideal_answer"]
                case_id = case.get("id", str(i))

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": question})

                # -- Step 1: call the target model --
                if model_provider == "nvidia_nim":
                    model_res = self._call_nvidia_nim(
                        model_name, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty
                    )
                elif model_provider == "openrouter":
                    model_res = self._call_openrouter(
                        model_name, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty
                    )
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
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "multi_turn_history_mode": multi_turn_history_mode,
            },
            "created_at": datetime.now().astimezone().isoformat(),
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


class ArenaRunner:
    """
    Runs the same dataset through multiple models simultaneously and compares
    them head-to-head. Each case gets a declared winner (or tie) based on LLM
    judge composite scores, and an optional direct pairwise comparison prompt.
    """

    def __init__(
        self,
        nvidia_nim_api_key: Optional[str] = None,
        nvidia_nim_base_url: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
    ):
        # Re-use the single EvaluationRunner for each contestant; keys are shared.
        self._base = EvaluationRunner(
            nvidia_nim_api_key=nvidia_nim_api_key,
            nvidia_nim_base_url=nvidia_nim_base_url,
            openrouter_api_key=openrouter_api_key,
        )

    # ------------------------------------------------------------------
    # Head-to-head judge
    # ------------------------------------------------------------------

    def _run_pairwise_judge(
        self,
        question: str,
        ideal_answer: str,
        responses: List[Dict[str, Any]],  # [{"model_label": str, "answer": str}, ...]
    ) -> Dict[str, Any]:
        """
        Asks an LLM to compare all contestant answers for a single question and
        pick the best one (or declare a tie). Returns {"winner_label": str, "reason": str}.
        Falls back to highest composite score when no LLM judge is available.
        """
        # Build the prompt dynamically for N contestants
        responses_text = "\n\n".join(
            f"### Response {i + 1} ({r['model_label']}):\n{r['answer']}"
            for i, r in enumerate(responses)
        )
        labels_str = ", ".join(r["model_label"] for r in responses)
        model_labels_json = json.dumps([r["model_label"] for r in responses])

        judge_prompt = f"""You are an impartial AI judge running a model arena evaluation.

Below is a user question, a golden reference answer, and {len(responses)} candidate responses from different models.
Your task is to select the BEST response overall, or declare a tie if two or more responses are equally strong.

Evaluation criteria (in order of importance):
1. Factual correctness vs the golden reference
2. Completeness — does it cover all key points?
3. Clarity and professional formatting

{responses_text}

---
User Question: {question}
Golden Reference Answer: {ideal_answer}
---

Respond ONLY with a valid JSON object following this exact schema:
{{
  "winner": "<model_label from this list: {labels_str}, or the string 'tie'>",
  "reason": "<1-2 sentence explanation citing the decisive difference>"
}}

JSON Response:"""

        def _parse(text: str) -> Optional[Dict[str, Any]]:
            try:
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                parsed = json.loads(text.strip())
                winner = parsed.get("winner", "tie")
                valid_labels = [r["model_label"] for r in responses] + ["tie"]
                if winner not in valid_labels:
                    winner = "tie"
                return {"winner": winner, "reason": parsed.get("reason", "")}
            except Exception:
                return None

        # 1. Try OpenRouter
        if self._base.openrouter_api_key:
            try:
                result = _parse(
                    self._base._call_openrouter(
                        "meta-llama/llama-3.2-3b-instruct",
                        [{"role": "user", "content": judge_prompt}],
                        temperature=0.1,
                    )["text"]
                )
                if result:
                    return result
            except Exception as e:
                print(f"ArenaRunner pairwise judge (OpenRouter) failed: {e}")

        # 2. Try Nvidia NIM
        if self._base.nvidia_nim_api_key:
            try:
                result = _parse(
                    self._base._call_nvidia_nim(
                        "meta/llama-3.2-3b-instruct",
                        [{"role": "user", "content": judge_prompt}],
                        temperature=0.1,
                    )["text"]
                )
                if result:
                    return result
            except Exception as e:
                print(f"ArenaRunner pairwise judge (Nvidia NIM) failed: {e}")

        # 3. Heuristic fallback — highest composite score wins
        best_score = -1.0
        best_label = "tie"
        tie_threshold = 0.1
        scores: List[tuple] = []
        for r in responses:
            score = (r.get("correctness", 0) + r.get("completeness", 0) + r.get("clarity", 0)) / 3.0
            scores.append((r["model_label"], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2 and abs(scores[0][1] - scores[1][1]) < tie_threshold:
            best_label = "tie"
            reason = "Heuristic fallback: composite scores too close to call — declared tie."
        else:
            best_label = scores[0][0] if scores else "tie"
            reason = f"Heuristic fallback: {best_label} achieved the highest composite judge score ({scores[0][1]:.2f}/5)."

        return {"winner": best_label, "reason": reason}

    # ------------------------------------------------------------------
    # Main arena pipeline
    # ------------------------------------------------------------------

    def run_arena(
        self,
        dataset: Dict[str, Any],
        arena_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Runs every contestant model over all dataset cases and produces a
        side-by-side comparison with per-case winners and aggregate leaderboard.

        arena_config shape:
          {
            "run_name": str,
            "dataset_id": str,
            "contestants": [
              {
                "model_provider": str,
                "model_name":     str,
                "temperature":    float,
                "system_prompt":  str,
                "max_tokens":     int | None,
                "top_p":          float | None,
                "frequency_penalty":  float | None,
                "presence_penalty":   float | None,
                "multi_turn_history_mode": str,
                "label":          str,   # display name, e.g. "Llama 3.1 8B"
              },
              ...
            ]
          }
        """
        contestants = arena_config.get("contestants", [])
        if len(contestants) < 2:
            raise ValueError("Arena requires at least 2 contestants.")

        cases = dataset.get("cases", [])
        total_cases = len(cases)

        # Per-contestant accumulators
        contestant_stats: Dict[str, Dict[str, Any]] = {}
        for c in contestants:
            label = c.get("label") or c.get("model_name", "unknown")
            contestant_stats[label] = {
                "model_provider": c.get("model_provider", "mock"),
                "model_name": c.get("model_name", "unknown"),
                "label": label,
                "wins": 0,
                "ties": 0,
                "losses": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
                "sum_correctness": 0.0,
                "sum_completeness": 0.0,
                "sum_clarity": 0.0,
                "sum_similarity": 0.0,
                "total_exact": 0,
            }

        arena_results: List[Dict[str, Any]] = []

        for case in cases:
            turns = case.get("turns")
            is_multi_turn = turns is not None and len(turns) > 0

            # ---- gather responses from each contestant for this case ----
            contestant_case_results: List[Dict[str, Any]] = []

            for c in contestants:
                label = c.get("label") or c.get("model_name", "unknown")
                provider = c.get("model_provider", "mock")
                model = c.get("model_name", "mock-model")
                temp = c.get("temperature", 0.2)
                sys_prompt = c.get("system_prompt", "")
                max_tokens = c.get("max_tokens")
                top_p = c.get("top_p")
                freq_pen = c.get("frequency_penalty")
                pres_pen = c.get("presence_penalty")
                mt_mode = c.get("multi_turn_history_mode", "model_response")

                if is_multi_turn:
                    # Run this contestant through all turns
                    turn_results = []
                    actual_history = []
                    case_lat = 0.0
                    case_cost = 0.0
                    case_exact = 0
                    case_sim = 0.0
                    case_corr = 0.0
                    case_comp = 0.0
                    case_clar = 0.0

                    for t_idx, turn in enumerate(turns):
                        user_msg = turn["user_message"]
                        ideal_resp = turn["ideal_response"]

                        messages = []
                        if sys_prompt:
                            messages.append({"role": "system", "content": sys_prompt})
                        for prev_t in range(t_idx):
                            messages.append({"role": "user", "content": turns[prev_t]["user_message"]})
                            hist_resp = (
                                actual_history[prev_t]
                                if mt_mode == "model_response"
                                else turns[prev_t]["ideal_response"]
                            )
                            messages.append({"role": "assistant", "content": hist_resp})
                        messages.append({"role": "user", "content": user_msg})

                        if provider == "nvidia_nim":
                            res = self._base._call_nvidia_nim(model, messages, temp, max_tokens, top_p, freq_pen, pres_pen)
                        elif provider == "openrouter":
                            res = self._base._call_openrouter(model, messages, temp, max_tokens, top_p, freq_pen, pres_pen)
                        else:
                            res = self._base._call_mock(model, user_msg, ideal_resp)

                        answer = res["text"]
                        actual_history.append(answer)

                        exact = self._base._get_exact_match(answer, ideal_resp)
                        sim = self._base._get_similarity_score(answer, ideal_resp)
                        judge = self._base._run_llm_as_judge(user_msg, ideal_resp, answer)
                        cost = self._base._calculate_costs(provider, model, res["input_tokens"], res["output_tokens"])

                        case_lat += res["latency"]
                        case_cost += cost
                        case_exact += exact
                        case_sim += sim
                        case_corr += judge["correctness"]
                        case_comp += judge["completeness"]
                        case_clar += judge["clarity"]

                        turn_results.append({
                            "turn_index": t_idx,
                            "user_message": user_msg,
                            "ideal_response": ideal_resp,
                            "model_response": answer,
                            "metrics": {
                                "exact_match": exact,
                                "similarity": sim,
                                "llm_correctness": judge["correctness"],
                                "llm_completeness": judge["completeness"],
                                "llm_clarity": judge["clarity"],
                                "latency": round(res["latency"], 2),
                                "cost": cost,
                                "input_tokens": res["input_tokens"],
                                "output_tokens": res["output_tokens"],
                                "reason": judge["reason"],
                            }
                        })

                    nt = len(turns)
                    contestant_case_results.append({
                        "label": label,
                        "model_provider": provider,
                        "model_name": model,
                        "is_multi_turn": True,
                        "turns": turn_results,
                        "model_answer": turn_results[0]["model_response"] if turn_results else "",
                        "correctness": round(case_corr / nt, 2),
                        "completeness": round(case_comp / nt, 2),
                        "clarity": round(case_clar / nt, 2),
                        "metrics": {
                            "exact_match": round(case_exact / nt, 2),
                            "similarity": round(case_sim / nt, 2),
                            "llm_correctness": round(case_corr / nt, 2),
                            "llm_completeness": round(case_comp / nt, 2),
                            "llm_clarity": round(case_clar / nt, 2),
                            "latency": round(case_lat, 2),
                            "cost": round(case_cost, 6),
                        }
                    })

                    # Accumulate stats
                    stats = contestant_stats[label]
                    stats["total_latency"] += case_lat
                    stats["total_cost"] += case_cost
                    stats["total_exact"] += case_exact / nt
                    stats["sum_correctness"] += case_corr / nt
                    stats["sum_completeness"] += case_comp / nt
                    stats["sum_clarity"] += case_clar / nt
                    stats["sum_similarity"] += case_sim / nt

                else:
                    # Single-turn
                    question = case["question"]
                    ideal_answer = case["ideal_answer"]

                    messages = []
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})
                    messages.append({"role": "user", "content": question})

                    if provider == "nvidia_nim":
                        res = self._base._call_nvidia_nim(model, messages, temp, max_tokens, top_p, freq_pen, pres_pen)
                    elif provider == "openrouter":
                        res = self._base._call_openrouter(model, messages, temp, max_tokens, top_p, freq_pen, pres_pen)
                    else:
                        res = self._base._call_mock(model, question, ideal_answer)

                    answer = res["text"]
                    exact = self._base._get_exact_match(answer, ideal_answer)
                    sim = self._base._get_similarity_score(answer, ideal_answer)
                    judge = self._base._run_llm_as_judge(question, ideal_answer, answer)
                    cost = self._base._calculate_costs(provider, model, res["input_tokens"], res["output_tokens"])

                    contestant_case_results.append({
                        "label": label,
                        "model_provider": provider,
                        "model_name": model,
                        "is_multi_turn": False,
                        "model_answer": answer,
                        "correctness": judge["correctness"],
                        "completeness": judge["completeness"],
                        "clarity": judge["clarity"],
                        "metrics": {
                            "exact_match": exact,
                            "similarity": sim,
                            "llm_correctness": judge["correctness"],
                            "llm_completeness": judge["completeness"],
                            "llm_clarity": judge["clarity"],
                            "latency": round(res["latency"], 2),
                            "cost": cost,
                            "input_tokens": res["input_tokens"],
                            "output_tokens": res["output_tokens"],
                            "reason": judge["reason"],
                        }
                    })

                    stats = contestant_stats[label]
                    stats["total_latency"] += res["latency"]
                    stats["total_cost"] += cost
                    stats["total_exact"] += exact
                    stats["sum_correctness"] += judge["correctness"]
                    stats["sum_completeness"] += judge["completeness"]
                    stats["sum_clarity"] += judge["clarity"]
                    stats["sum_similarity"] += sim

            # ---- pairwise judge for this case ----
            pairwise_inputs = [
                {
                    "model_label": r["label"],
                    "answer": r["model_answer"],
                    "correctness": r.get("correctness", 0),
                    "completeness": r.get("completeness", 0),
                    "clarity": r.get("clarity", 0),
                }
                for r in contestant_case_results
            ]
            question_text = (
                turns[0]["user_message"] if is_multi_turn else case.get("question", "")
            )
            ideal_text = (
                turns[0]["ideal_response"] if is_multi_turn else case.get("ideal_answer", "")
            )
            pairwise = self._run_pairwise_judge(question_text, ideal_text, pairwise_inputs)
            case_winner = pairwise["winner"]
            case_winner_reason = pairwise["reason"]

            # Update win/tie/loss counts
            for r in contestant_case_results:
                lbl = r["label"]
                if case_winner == "tie":
                    contestant_stats[lbl]["ties"] += 1
                elif case_winner == lbl:
                    contestant_stats[lbl]["wins"] += 1
                else:
                    contestant_stats[lbl]["losses"] += 1

            arena_results.append({
                "case_id": case.get("id", ""),
                "is_multi_turn": is_multi_turn,
                "question": question_text,
                "ideal_answer": ideal_text,
                "contestant_results": contestant_case_results,
                "winner": case_winner,
                "winner_reason": case_winner_reason,
            })

        # ---- build aggregate per-contestant metrics ----
        n = total_cases or 1
        leaderboard: List[Dict[str, Any]] = []
        for label, stats in contestant_stats.items():
            win_rate = round(stats["wins"] / total_cases, 3) if total_cases else 0.0
            leaderboard.append({
                "label": label,
                "model_provider": stats["model_provider"],
                "model_name": stats["model_name"],
                "wins": stats["wins"],
                "ties": stats["ties"],
                "losses": stats["losses"],
                "win_rate": win_rate,
                "avg_correctness": round(stats["sum_correctness"] / n, 2),
                "avg_completeness": round(stats["sum_completeness"] / n, 2),
                "avg_clarity": round(stats["sum_clarity"] / n, 2),
                "avg_similarity": round(stats["sum_similarity"] / n, 3),
                "avg_latency": round(stats["total_latency"] / n, 2),
                "total_cost": round(stats["total_cost"], 6),
            })
        leaderboard.sort(key=lambda x: (x["wins"], x["avg_correctness"]), reverse=True)

        return {
            "type": "arena",
            "name": arena_config.get("run_name", f"Arena Run {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
            "dataset_id": dataset["id"],
            "dataset_name": dataset["name"],
            "created_at": datetime.now().astimezone().isoformat(),
            "total_cases": total_cases,
            "contestants": [
                {
                    "label": c.get("label") or c.get("model_name"),
                    "model_provider": c.get("model_provider"),
                    "model_name": c.get("model_name"),
                    "parameters": {
                        "temperature": c.get("temperature", 0.2),
                        "system_prompt": c.get("system_prompt", ""),
                        "max_tokens": c.get("max_tokens"),
                        "top_p": c.get("top_p"),
                        "frequency_penalty": c.get("frequency_penalty"),
                        "presence_penalty": c.get("presence_penalty"),
                        "multi_turn_history_mode": c.get("multi_turn_history_mode", "model_response"),
                    }
                }
                for c in contestants
            ],
            "leaderboard": leaderboard,
            "results": arena_results,
        }
