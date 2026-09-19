"""
Evaluation instrumentation for research revision smoke tests and benchmarks.

Provides run_id, stage events, failure taxonomy logging, LLM usage logs,
human-intervention logs, manifests, and result.json writers.

Does not change core ML methods — only observability and eval-mode policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Failure codes aligned with research_revision/FAILURE_TAXONOMY.md
FAILURE_CODES = {
    "F01": "LLM reasoning failure",
    "F02": "Intent classification failure",
    "F03": "Ambiguous user request",
    "F04": "Dataset discovery failure",
    "F05": "Dataset access failure",
    "F06": "Dataset schema mismatch",
    "F07": "Preprocessing failure",
    "F08": "AutoML failure",
    "F09": "Model training failure",
    "F10": "Validation failure",
    "F11": "Code-generation failure",
    "F12": "Repository-generation failure",
    "F13": "Deployment / publish failure",
    "F14": "Infrastructure failure",
    "F15": "Timeout",
    "F16": "Resource exhaustion",
    "F17": "Synthetic fallback used",
    "F18": "Forced weak match",
    "F19": "Data leakage",
    "F20": "Human intervention",
}

FAILURE_CATEGORY_MAP = {
    "F01": "formulation failure",
    "F02": "intent failure",
    "F03": "formulation failure",
    "F04": "dataset discovery failure",
    "F05": "dataset access failure",
    "F06": "schema mismatch",
    "F07": "preprocessing failure",
    "F08": "model training failure",
    "F09": "model training failure",
    "F10": "validation failure",
    "F11": "code generation failure",
    "F12": "repository failure",
    "F13": "deployment/publication failure",
    "F14": "infrastructure failure",
    "F15": "timeout",
    "F16": "resource exhaustion",
    "F17": "dataset discovery failure",
    "F18": "dataset discovery failure",
    "F19": "validation failure",
    "F20": "human intervention",
}

# Canonical stage names used in evaluation traces (map to real pipeline stages)
STAGE_INPUT = "INPUT"
STAGE_INTENT = "INTENT_CLASSIFICATION"
STAGE_FORMULATION = "PROBLEM_FORMULATION"
STAGE_PROBLEM_VALIDATION = "PROBLEM_VALIDATION"
STAGE_DATASET_DISCOVERY = "DATASET_DISCOVERY"
STAGE_DATASET_VALIDATION = "DATASET_VALIDATION"
STAGE_PREPROCESSING = "PREPROCESSING"
STAGE_MODEL_SEARCH = "MODEL_SEARCH"
STAGE_MODEL_TRAINING = "MODEL_TRAINING"
STAGE_MODEL_VALIDATION = "MODEL_VALIDATION"
STAGE_CODE_GENERATION = "CODE_GENERATION"
STAGE_CODE_VALIDATION = "CODE_VALIDATION"
STAGE_REPOSITORY_GENERATION = "REPOSITORY_GENERATION"
STAGE_REPOSITORY_VALIDATION = "REPOSITORY_VALIDATION"
STAGE_DEPLOYMENT = "DEPLOYMENT_PUBLICATION"
STAGE_FINALIZATION = "FINALIZATION"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    try:
        cwd = str(repo_root) if repo_root else None
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def config_hash(config: Dict[str, Any]) -> str:
    """Hash config with secrets redacted."""
    redacted = _redact_secrets(config)
    payload = json.dumps(redacted, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _redact_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key_l = str(k).lower()
            if any(s in key_l for s in ("token", "password", "secret", "api_key", "apikey")):
                out[k] = "***REDACTED***"
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(x) for x in obj]
    return obj


@dataclass
class StageEvent:
    run_id: str
    task_id: str
    seed: Optional[int]
    stage: str
    status: str  # SUCCESS | FAILURE | SKIPPED
    start_time: str
    end_time: str
    duration_seconds: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureRecord:
    run_id: str
    task_id: str
    stage: str
    failure_code: str
    failure_category: str
    exception_type: Optional[str]
    message: str
    recoverable: bool
    retry_attempted: bool
    retry_result: Optional[str]
    downstream_stages_affected: List[str]
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMCallRecord:
    run_id: str
    provider: str
    model: str
    purpose_stage: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    token_count_status: str  # measured | unavailable
    latency_seconds: float
    estimated_cost_usd: Optional[float]
    success: bool
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HumanInterventionRecord:
    run_id: str
    required: bool
    stage: Optional[str] = None
    intervention_type: Optional[str] = None
    reason: Optional[str] = None
    action_taken: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvalRunRecorder:
    """Collects structured evidence for one pipeline evaluation run."""

    def __init__(
        self,
        output_root: Path,
        task_id: str,
        seed: int,
        evaluation_config: Dict[str, Any],
        full_config: Dict[str, Any],
        system_version: str = "0.1.0-eval",
        dataset_version: str = "benchmark-v1-smoke",
        repo_root: Optional[Path] = None,
    ):
        self.run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.task_id = task_id
        self.seed = seed
        self.evaluation_config = evaluation_config or {}
        self.full_config = full_config or {}
        self.system_version = system_version
        self.dataset_version = dataset_version
        self.repo_root = repo_root
        self.output_dir = Path(output_root) / "runs" / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stage_events: List[StageEvent] = []
        self.failures: List[FailureRecord] = []
        self.llm_calls: List[LLMCallRecord] = []
        self.human_interventions: List[HumanInterventionRecord] = []
        self.artifacts: Dict[str, Any] = {
            "problem": {},
            "dataset": {},
            "training": {},
            "validation": {},
            "code_generation": {},
            "repository": {},
            "deployment": {},
            "gate_decisions": {},
            "timings": {},
        }
        self.final_status = "FAILURE"
        self._stage_starts: Dict[str, datetime] = {}
        self.created_at = utc_now_iso()

        # Default: no human intervention unless recorded
        self.human_interventions.append(
            HumanInterventionRecord(run_id=self.run_id, required=False)
        )

    # --- policy helpers ---
    def eval_enabled(self) -> bool:
        return bool(self.evaluation_config.get("enabled", False))

    def gates_enabled(self) -> bool:
        gates = self.evaluation_config.get("gates", {})
        if isinstance(gates, dict):
            return bool(gates.get("enabled", True)) if self.eval_enabled() else False
        return bool(gates)

    def synthetic_allowed(self) -> bool:
        if not self.eval_enabled():
            # development default: allow unless explicitly disabled in eval block
            syn = self.evaluation_config.get("synthetic_data", {})
            if isinstance(syn, dict) and "enabled" in syn:
                return bool(syn.get("enabled"))
            return True
        syn = self.evaluation_config.get("synthetic_data", {})
        if isinstance(syn, dict):
            return bool(syn.get("enabled", False))
        return False

    def retries_enabled(self) -> bool:
        retries = self.evaluation_config.get("retries", {})
        if isinstance(retries, dict):
            return bool(retries.get("enabled", False))
        return False

    # --- stage lifecycle ---
    def begin_stage(self, stage: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._stage_starts[stage] = datetime.now(timezone.utc)
        logger.info(f"[EVAL] BEGIN stage={stage} run_id={self.run_id}")
        if metadata:
            self.artifacts.setdefault("timings", {})
            # store pending metadata on start via temporary key
            self.artifacts.setdefault("_pending_stage_meta", {})[stage] = metadata

    def end_stage(
        self,
        stage: str,
        status: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StageEvent:
        start = self._stage_starts.pop(stage, datetime.now(timezone.utc))
        end = datetime.now(timezone.utc)
        duration = (end - start).total_seconds()
        pending = self.artifacts.get("_pending_stage_meta", {}).pop(stage, {})
        meta = {**(pending or {}), **(metadata or {})}
        event = StageEvent(
            run_id=self.run_id,
            task_id=self.task_id,
            seed=self.seed,
            stage=stage,
            status=status,
            start_time=start.isoformat(),
            end_time=end.isoformat(),
            duration_seconds=duration,
            error_type=error_type,
            error_message=error_message,
            metadata=meta,
        )
        self.stage_events.append(event)
        self.artifacts["timings"][stage] = duration
        self._write_json("stage_events.jsonl", None, append_line=event.to_dict())
        logger.info(
            f"[EVAL] END stage={stage} status={status} duration={duration:.3f}s run_id={self.run_id}"
        )
        return event

    @contextmanager
    def stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.begin_stage(stage_name, metadata)
        try:
            yield
            self.end_stage(stage_name, "SUCCESS")
        except Exception as e:
            self.end_stage(
                stage_name,
                "FAILURE",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    def skip_stage(self, stage: str, reason: str) -> StageEvent:
        now = utc_now_iso()
        event = StageEvent(
            run_id=self.run_id,
            task_id=self.task_id,
            seed=self.seed,
            stage=stage,
            status="SKIPPED",
            start_time=now,
            end_time=now,
            duration_seconds=0.0,
            metadata={"reason": reason},
        )
        self.stage_events.append(event)
        self._write_json("stage_events.jsonl", None, append_line=event.to_dict())
        return event

    def record_failure(
        self,
        stage: str,
        failure_code: str,
        message: str,
        exception: Optional[BaseException] = None,
        recoverable: bool = False,
        retry_attempted: bool = False,
        retry_result: Optional[str] = None,
        downstream_stages_affected: Optional[List[str]] = None,
    ) -> FailureRecord:
        category = FAILURE_CATEGORY_MAP.get(failure_code, "infrastructure failure")
        rec = FailureRecord(
            run_id=self.run_id,
            task_id=self.task_id,
            stage=stage,
            failure_code=failure_code,
            failure_category=category,
            exception_type=type(exception).__name__ if exception else None,
            message=message,
            recoverable=recoverable,
            retry_attempted=retry_attempted,
            retry_result=retry_result,
            downstream_stages_affected=downstream_stages_affected
            or [
                STAGE_MODEL_TRAINING,
                STAGE_CODE_GENERATION,
                STAGE_DEPLOYMENT,
            ],
        )
        self.failures.append(rec)
        self._write_json("failures.jsonl", None, append_line=rec.to_dict())
        logger.warning(
            f"[EVAL] FAILURE {failure_code} ({category}) stage={stage}: {message}"
        )
        return rec

    def record_llm_call(
        self,
        provider: str,
        model: str,
        purpose_stage: str,
        latency_seconds: float,
        success: bool,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> LLMCallRecord:
        if input_tokens is None and output_tokens is None and total_tokens is None:
            token_status = "unavailable"
        else:
            token_status = "measured"
        rec = LLMCallRecord(
            run_id=self.run_id,
            provider=provider,
            model=model,
            purpose_stage=purpose_stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_count_status=token_status,
            latency_seconds=latency_seconds,
            estimated_cost_usd=estimated_cost_usd,
            success=success,
            error_message=error_message,
        )
        self.llm_calls.append(rec)
        self._write_json("llm_calls.jsonl", None, append_line=rec.to_dict())
        return rec

    def record_human_intervention(
        self,
        required: bool,
        stage: Optional[str] = None,
        intervention_type: Optional[str] = None,
        reason: Optional[str] = None,
        action_taken: Optional[str] = None,
    ) -> HumanInterventionRecord:
        # Replace default false-only record if we get a real intervention
        if required:
            self.human_interventions = [
                h for h in self.human_interventions if h.required
            ]
        rec = HumanInterventionRecord(
            run_id=self.run_id,
            required=required,
            stage=stage,
            intervention_type=intervention_type,
            reason=reason,
            action_taken=action_taken,
        )
        if required or not any(not h.required for h in self.human_interventions):
            self.human_interventions.append(rec)
        self._write_json("human_intervention.jsonl", None, append_line=rec.to_dict())
        return rec

    def set_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def finalize(self, status: str) -> Dict[str, Any]:
        self.final_status = status
        if not any(e.stage == STAGE_FINALIZATION for e in self.stage_events):
            self.begin_stage(STAGE_FINALIZATION)
            self.end_stage(STAGE_FINALIZATION, "SUCCESS")

        manifest = self._build_manifest(status)
        result = self._build_result(status)
        self._write_json("manifest.json", manifest)
        self._write_json("result.json", result)
        self._write_json("stage_trace.json", [e.to_dict() for e in self.stage_events])
        return result

    def _build_manifest(self, status: str) -> Dict[str, Any]:
        eval_cfg = self.evaluation_config
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "evaluation_mode": bool(eval_cfg.get("enabled", False)),
            "gates_enabled": self.gates_enabled(),
            "synthetic_data_enabled": self.synthetic_allowed(),
            "retries_enabled": self.retries_enabled(),
            "system_version": self.system_version,
            "git_commit": get_git_commit(self.repo_root),
            "timestamp": self.created_at,
            "dataset_version": self.dataset_version,
            "config_hash": config_hash(self.full_config),
            "status": status,
            "output_dir": str(self.output_dir),
        }

    def _build_result(self, status: str) -> Dict[str, Any]:
        hi_required = any(h.required for h in self.human_interventions)
        return {
            "run": self._build_manifest(status),
            "problem": self.artifacts.get("problem", {}),
            "dataset": self.artifacts.get("dataset", {}),
            "training": self.artifacts.get("training", {}),
            "validation": self.artifacts.get("validation", {}),
            "code_generation": self.artifacts.get("code_generation", {}),
            "repository": self.artifacts.get("repository", {}),
            "deployment": self.artifacts.get("deployment", {}),
            "gate_decisions": self.artifacts.get("gate_decisions", {}),
            "llm_usage": {
                "calls": [c.to_dict() for c in self.llm_calls],
                "n_calls": len(self.llm_calls),
            },
            "human_intervention": {
                "required": hi_required,
                "records": [h.to_dict() for h in self.human_interventions],
            },
            "failures": [f.to_dict() for f in self.failures],
            "timings": self.artifacts.get("timings", {}),
            "stage_trace": [e.to_dict() for e in self.stage_events],
            "final_status": {
                "status": status,
                "success": status == "SUCCESS",
                "n_failures": len(self.failures),
                "n_stages": len(self.stage_events),
            },
        }

    def _write_json(
        self,
        name: str,
        data: Any,
        append_line: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = self.output_dir / name
        if append_line is not None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(append_line, default=str) + "\n")
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


def load_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize evaluation_mode block from config."""
    raw = config.get("evaluation_mode") or config.get("evaluation") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "gates": {
            "enabled": bool((raw.get("gates") or {}).get("enabled", True))
            if isinstance(raw.get("gates"), dict)
            else bool(raw.get("gates", True))
        },
        "synthetic_data": {
            "enabled": bool((raw.get("synthetic_data") or {}).get("enabled", False))
            if isinstance(raw.get("synthetic_data"), dict)
            else bool(raw.get("synthetic_data", False))
        },
        "retries": {
            "enabled": bool((raw.get("retries") or {}).get("enabled", False))
            if isinstance(raw.get("retries"), dict)
            else bool(raw.get("retries", False)),
            "max_retries": int((raw.get("retries") or {}).get("max_retries", 0))
            if isinstance(raw.get("retries"), dict)
            else 0,
        },
        "logging": {
            "stage_level": True,
            "token_level": True,
            "failure_level": True,
            "human_intervention": True,
            **(raw.get("logging") or {}),
        },
        "output_root": raw.get(
            "output_root",
            str(Path(__file__).resolve().parents[2] / "research_revision" / "results"),
        ),
        "dataset_version": raw.get("dataset_version", "benchmark-v1-smoke"),
        "force_all_gates_in_direct_mode": bool(
            raw.get("force_all_gates_in_direct_mode", True)
        ),
    }
