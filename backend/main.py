import os
import csv
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
from urllib.parse import urlparse

from database import Database
from evaluator import EvaluationRunner, ArenaRunner

app = FastAPI(title="LLM Evaluation Starter Framework API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()

# Pydantic schemas
class ModelConfig(BaseModel):
    run_name: str = Field(..., json_schema_extra={"example": "Llama 3.2 on Reasoning Benchmark"})
    dataset_id: str = Field(..., json_schema_extra={"example": "logical_reasoning_benchmark"})
    model_provider: str = Field(..., json_schema_extra={"example": "openrouter"})  # mock | nvidia_nim | openrouter
    model_name: str = Field(..., json_schema_extra={"example": "meta-llama/llama-3.1-8b-instruct:free"})
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    system_prompt: Optional[str] = ""
    # Exposing more parameters
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    # Multi-turn history evaluation mode (model_response or ideal_response)
    multi_turn_history_mode: str = Field("model_response", json_schema_extra={"example": "model_response"})
    # Per-provider credentials (fall back to env vars when omitted)
    nvidia_nim_api_key: Optional[str] = None
    nvidia_nim_base_url: Optional[str] = None
    openrouter_api_key: Optional[str] = None

class ManualTurn(BaseModel):
    user_message: str
    ideal_response: str

class ManualCase(BaseModel):
    id: Optional[str] = None
    question: Optional[str] = None
    ideal_answer: Optional[str] = None
    turns: Optional[List[ManualTurn]] = None

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    cases: List[ManualCase]

class ContestantConfig(BaseModel):
    label: Optional[str] = None           # display name; falls back to model_name
    model_provider: str = Field(..., json_schema_extra={"example": "openrouter"})
    model_name: str = Field(..., json_schema_extra={"example": "meta-llama/llama-3.1-8b-instruct:free"})
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    system_prompt: Optional[str] = ""
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    multi_turn_history_mode: str = Field("model_response")

class ArenaConfig(BaseModel):
    run_name: str = Field(..., json_schema_extra={"example": "Llama 3.1 8B vs Mistral 7B Arena"})
    dataset_id: str = Field(..., json_schema_extra={"example": "logical_reasoning_benchmark"})
    contestants: List[ContestantConfig] = Field(..., min_length=2)
    # Shared credentials (fall back to env vars when omitted)
    nvidia_nim_api_key: Optional[str] = None
    nvidia_nim_base_url: Optional[str] = None
    openrouter_api_key: Optional[str] = None

class HFImportRequest(BaseModel):
    path: str
    config: Optional[str] = None
    split: str
    question_column: str
    answer_column: str
    choices_column: Optional[str] = None
    limit: int = 50
    dataset_name: str
    dataset_description: Optional[str] = ""

@app.get("/api/check-keys")
async def check_keys():
    """Checks which API keys are pre-configured as environment variables."""
    return {
        "nvidia_nim_api_key_set": bool(os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")),
        "nvidia_nim_base_url_set": bool(os.environ.get("NVIDIA_NIM_API_BASE") or os.environ.get("NVIDIA_API_BASE")),
        "openrouter_api_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        "nvidia_nim_base_url": os.environ.get("NVIDIA_NIM_API_BASE") or os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"),
    }

# Dataset endpoints
@app.get("/api/datasets")
async def list_datasets():
    return db.get_datasets()

# HuggingFace Datasets Integration endpoints
@app.get("/api/datasets/preview-hf")
async def preview_hf(path: str):
    """Fetches structure metadata and preview rows from a Hugging Face Hub dataset."""
    try:
        # 1. Fetch available configurations (subsets)
        try:
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names(path)
        except Exception:
            configs = []
            
        config_name = configs[0] if configs else None
        
        # 2. Fetch available splits
        try:
            from datasets import get_dataset_split_names
            splits = get_dataset_split_names(path, config_name=config_name)
        except Exception:
            splits = ["train", "validation", "test"]
            
        split_name = "validation" if "validation" in splits else ("test" if "test" in splits else (splits[0] if splits else "train"))
        
        # 3. Stream a small preview of rows
        from datasets import load_dataset
        preview_rows = []
        try:
            ds = load_dataset(path, name=config_name, split=split_name, streaming=True)
            iterator = iter(ds)
            for _ in range(3):
                try:
                    preview_rows.append(next(iterator))
                except StopIteration:
                    break
        except Exception:
            # Fallback to standard full load (slower, but covers datasets without streaming support)
            try:
                ds = load_dataset(path, name=config_name, split=split_name)
                preview_rows = [ds[i] for i in range(min(3, len(ds)))]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not load Hugging Face dataset: {str(e)}")
                
        if not preview_rows:
            raise HTTPException(status_code=400, detail="Hugging Face dataset has no valid records.")
            
        columns = list(preview_rows[0].keys())
        
        return {
            "configs": configs,
            "splits": splits,
            "columns": columns,
            "preview_rows": preview_rows
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to preview HuggingFace dataset '{path}': {str(e)}")

@app.post("/api/datasets/import-hf")
async def import_hf(req: HFImportRequest):
    """Downloads rows from Hugging Face, applies mapped schema, and stores it locally."""
    try:
        from datasets import load_dataset
        
        # Load dataset
        try:
            ds = load_dataset(req.path, name=req.config, split=req.split, streaming=True)
            iterator = iter(ds)
            rows = []
            for _ in range(req.limit):
                try:
                    rows.append(next(iterator))
                except StopIteration:
                    break
        except Exception:
            # Non-streaming fallback
            ds = load_dataset(req.path, name=req.config, split=req.split)
            rows = [ds[i] for i in range(min(req.limit, len(ds)))]
            
        if not rows:
            raise HTTPException(status_code=400, detail="Hugging Face dataset holds no records under selected config/split.")
            
        cases = []
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        for idx, row in enumerate(rows):
            question = str(row.get(req.question_column) or "").strip()
            
            # Retrieve choices for multiple choice QAs
            choices = None
            if req.choices_column:
                choices = row.get(req.choices_column)
                
            raw_answer = row.get(req.answer_column)
            
            # Map choice index to string text
            if choices and isinstance(choices, list) and len(choices) > 0:
                formatted_choices = "\n".join([
                    f"{choice_labels[j]}. {choice_text}" 
                    for j, choice_text in enumerate(choices) 
                    if j < len(choice_labels)
                ])
                question = f"{question}\n\nChoices:\n{formatted_choices}"
                
                try:
                    ans_idx = int(raw_answer)
                    if 0 <= ans_idx < len(choices):
                        ideal_answer = f"{choice_labels[ans_idx]}. {choices[ans_idx]}"
                    else:
                        ideal_answer = str(raw_answer)
                except (ValueError, TypeError):
                    # Answer is already choice option code 'A', 'B', etc.
                    raw_str = str(raw_answer).strip().upper()
                    if raw_str in choice_labels:
                        lbl_idx = choice_labels.index(raw_str)
                        if lbl_idx < len(choices):
                            ideal_answer = f"{raw_str}. {choices[lbl_idx]}"
                        else:
                            ideal_answer = raw_str
                    else:
                        ideal_answer = str(raw_answer)
            else:
                # Text answers (like MS MARCO answers)
                if isinstance(raw_answer, list):
                    ideal_answer = next((str(x) for x in raw_answer if x), "")
                else:
                    ideal_answer = str(raw_answer)
                    
            if question and ideal_answer:
                cases.append({
                    "id": f"hf-case-{idx+1}",
                    "question": question,
                    "ideal_answer": ideal_answer
                })
                
        if not cases:
            raise HTTPException(status_code=400, detail="Column mapping returned 0 valid cases. Verify column headers.")
            
        dataset = {
            "name": req.dataset_name,
            "description": req.dataset_description or f"Imported from Hugging Face: {req.path}",
            "cases": cases
        }
        saved = db.save_dataset(dataset)
        return saved
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to import dataset: {str(e)}")

@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@app.post("/api/datasets")
async def create_dataset(payload: DatasetCreate):
    if not payload.name:
        raise HTTPException(status_code=400, detail="Dataset name is required")
    if not payload.cases:
        raise HTTPException(status_code=400, detail="Dataset must contain at least one case")
        
    cases = []
    for i, c in enumerate(payload.cases):
        if c.turns:
            cases.append({
                "id": c.id or f"case-{i+1}",
                "turns": [{"user_message": t.user_message, "ideal_response": t.ideal_response} for t in c.turns]
            })
        else:
            cases.append({
                "id": c.id or f"case-{i+1}",
                "question": c.question,
                "ideal_answer": c.ideal_answer
            })
            
    dataset = {
        "name": payload.name,
        "description": payload.description,
        "cases": cases
    }
    return db.save_dataset(dataset)

@app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...)
):
    contents = await file.read()
    filename = file.filename.lower()
    cases = []

    try:
        if filename.endswith(".json"):
            data = json.loads(contents.decode("utf-8"))
            if isinstance(data, list):
                raw_cases = data
            elif isinstance(data, dict) and "cases" in data:
                raw_cases = data["cases"]
            else:
                raise ValueError("JSON must be a list of Q&A cases or an object containing a 'cases' list.")
            
            for i, item in enumerate(raw_cases):
                turns = item.get("turns")
                q = item.get("question") or item.get("prompt")
                a = item.get("ideal_answer") or item.get("reference") or item.get("answer")
                if not turns and (not q or not a):
                    raise ValueError(f"Case {i+1} is missing 'question' or 'ideal_answer' or 'turns'")
                
                if turns:
                    for t_idx, turn in enumerate(turns):
                        if "user_message" not in turn or "ideal_response" not in turn:
                            raise ValueError(f"Case {i+1} Turn {t_idx+1} is missing 'user_message' or 'ideal_response'")
                    cases.append({
                        "id": item.get("id", f"case-{i+1}"),
                        "turns": [{"user_message": t["user_message"], "ideal_response": t["ideal_response"]} for t in turns]
                    })
                else:
                    cases.append({
                        "id": item.get("id", f"case-{i+1}"),
                        "question": q,
                        "ideal_answer": a
                    })
                
        elif filename.endswith(".csv"):
            decoded = contents.decode("utf-8")
            reader = csv.DictReader(io.StringIO(decoded))
            
            fieldnames = reader.fieldnames or []
            question_key = None
            answer_key = None
            
            for f in fieldnames:
                fl = f.lower().strip()
                if fl in ["question", "prompt", "query", "input"]:
                    question_key = f
                if fl in ["ideal_answer", "answer", "reference", "target", "gold"]:
                    answer_key = f
                    
            if not question_key or not answer_key:
                if len(fieldnames) >= 2:
                    question_key = fieldnames[0]
                    answer_key = fieldnames[1]
                else:
                    raise ValueError("CSV must have at least two columns for question and ideal answer.")
                    
            for i, row in enumerate(reader):
                q = row.get(question_key)
                a = row.get(answer_key)
                if q and a:
                    cases.append({
                        "id": row.get("id", f"case-{i+1}"),
                        "question": q.strip(),
                        "ideal_answer": a.strip()
                    })
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .json or .csv")

        if not cases:
            raise ValueError("No valid Q&A cases found in the uploaded file.")

        dataset = {
            "name": name,
            "description": description,
            "cases": cases
        }
        return db.save_dataset(dataset)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset file: {str(e)}")



@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    success = db.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "success", "message": "Dataset deleted successfully"}

# Evaluation Runs endpoints
@app.get("/api/runs")
async def list_runs():
    return db.get_runs()

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run

@app.post("/api/runs")
async def execute_run(config: ModelConfig):
    # Fetch dataset
    dataset = db.get_dataset(config.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Resolve keys: prefer request payload, fall back to env vars
    nvidia_nim_key       = config.nvidia_nim_api_key or os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    nvidia_nim_base_url  = config.nvidia_nim_base_url or os.environ.get("NVIDIA_NIM_API_BASE") or os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    openrouter_key       = config.openrouter_api_key    or os.environ.get("OPENROUTER_API_KEY")

    # Provider-specific validation
    if config.model_provider == "nvidia_nim":
        parsed_base_url = urlparse(nvidia_nim_base_url)
        is_default_base = parsed_base_url.hostname == "integrate.api.nvidia.com"
        if is_default_base and not nvidia_nim_key:
            raise HTTPException(status_code=400, detail="Nvidia NIM API key is required when using the default cloud host. Set NVIDIA_NIM_API_KEY or provide it in the request.")
    if config.model_provider == "openrouter" and not openrouter_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key is required. Set OPENROUTER_API_KEY or provide it in the request.")

    runner = EvaluationRunner(
        nvidia_nim_api_key=nvidia_nim_key,
        nvidia_nim_base_url=nvidia_nim_base_url,
        openrouter_api_key=openrouter_key,
    )

    try:
        run_data = runner.run_evaluation(dataset, config.dict())
        saved_run = db.save_run(run_data)
        return saved_run
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    success = db.delete_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "success", "message": "Evaluation run deleted successfully"}


# ---------------------------------------------------------------------------
# Arena endpoints
# ---------------------------------------------------------------------------

@app.get("/api/arena-runs")
async def list_arena_runs():
    """Returns summary list of all saved arena runs (no per-case results)."""
    return db.get_arena_runs()


@app.get("/api/arena-runs/{run_id}")
async def get_arena_run(run_id: str):
    run = db.get_arena_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Arena run not found")
    return run


@app.post("/api/arena-runs")
async def execute_arena_run(config: ArenaConfig):
    dataset = db.get_dataset(config.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Resolve shared credentials
    nvidia_nim_key      = config.nvidia_nim_api_key  or os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    nvidia_nim_base_url = config.nvidia_nim_base_url or os.environ.get("NVIDIA_NIM_API_BASE") or os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    openrouter_key      = config.openrouter_api_key  or os.environ.get("OPENROUTER_API_KEY")

    # Provider-specific validation across all contestants
    for c in config.contestants:
        if c.model_provider == "nvidia_nim":
            is_default_base = "integrate.api.nvidia.com" in (nvidia_nim_base_url or "")
            if is_default_base and not nvidia_nim_key:
                raise HTTPException(
                    status_code=400,
                    detail=f"Nvidia NIM API key is required for contestant '{c.label or c.model_name}'. Set NVIDIA_NIM_API_KEY or provide it in the request."
                )
        if c.model_provider == "openrouter" and not openrouter_key:
            raise HTTPException(
                status_code=400,
                detail=f"OpenRouter API key is required for contestant '{c.label or c.model_name}'. Set OPENROUTER_API_KEY or provide it in the request."
            )

    runner = ArenaRunner(
        nvidia_nim_api_key=nvidia_nim_key,
        nvidia_nim_base_url=nvidia_nim_base_url,
        openrouter_api_key=openrouter_key,
    )

    # Merge label defaults and build config dict
    contestants_dicts = []
    for c in config.contestants:
        d = c.dict()
        if not d.get("label"):
            d["label"] = d["model_name"]
        contestants_dicts.append(d)

    arena_cfg = {
        "run_name": config.run_name,
        "dataset_id": config.dataset_id,
        "contestants": contestants_dicts,
    }

    try:
        run_data = runner.run_arena(dataset, arena_cfg)
        saved = db.save_arena_run(run_data)
        return saved
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arena evaluation failed: {str(e)}")


@app.delete("/api/arena-runs/{run_id}")
async def delete_arena_run(run_id: str):
    success = db.delete_arena_run(run_id)
    if not success:
        raise HTTPException(status_code=404, detail="Arena run not found")
    return {"status": "success", "message": "Arena run deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
