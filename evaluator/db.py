"""
evaluator.db
============

SQLite database interface for storing and querying evaluation results.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class EvalDB:
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT,
                    discovered_models_json TEXT
                )
            ''')
            
            # 2. Raw responses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model_name TEXT,
                    prompt_id TEXT,
                    category TEXT,
                    prompt_text TEXT,
                    response_text TEXT,
                    success BOOLEAN,
                    error TEXT,
                    latency_ms FLOAT,
                    raw_metadata_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            ''')
            
            # 3. Scored responses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scored_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model_name TEXT,
                    prompt_id TEXT,
                    metric_name TEXT,
                    value FLOAT,
                    rationale TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            ''')
            
            # 4. Aggregated summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aggregated_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    model_name TEXT,
                    category TEXT,
                    mean_score FLOAT,
                    safety_score FLOAT,
                    mean_latency_ms FLOAT,
                    median_latency_ms FLOAT,
                    p95_latency_ms FLOAT,
                    tokens_per_sec FLOAT,
                    narrative_summary TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            ''')
            
            conn.commit()

    def save_run(self, run_id: str, config: Dict, discovered_models: List[str]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO runs (run_id, config_json, discovered_models_json) VALUES (?, ?, ?)",
                (run_id, json.dumps(config), json.dumps(discovered_models))
            )
            conn.commit()

    def save_raw_response(self, run_id: str, model_name: str, prompt_id: str, 
                          category: str, prompt_text: str, response_text: str, success: bool, 
                          error: Optional[str], latency_ms: float, raw_metadata: Dict):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO raw_responses 
                (run_id, model_name, prompt_id, category, prompt_text, response_text, success, error, latency_ms, raw_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (run_id, model_name, prompt_id, category, prompt_text, response_text, success, error, latency_ms, json.dumps(raw_metadata)))
            conn.commit()

    def save_scores(self, run_id: str, model_name: str, prompt_id: str, scores: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for metric, data in scores.items():
                if metric in ["error", "llm_judge"]:
                    continue
                
                # Handle nested dicts from scorers (e.g., {"score": 1.0, "rationale": "..."})
                # and simple values if they exist
                if isinstance(data, dict):
                    value = data.get(metric, data.get("score"))
                    # If we can't find a numeric value, skip or try to convert
                    if value is None:
                         # try other keys
                         for k, v in data.items():
                             if isinstance(v, (int, float, bool)):
                                 value = float(v)
                                 break
                    else:
                        value = float(value)
                    rationale = data.get("rationale", "")
                else:
                    try:
                        value = float(data)
                        rationale = ""
                    except:
                        continue
                
                if value is not None:
                    cursor.execute('''
                        INSERT INTO scored_responses (run_id, model_name, prompt_id, metric_name, value, rationale)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (run_id, model_name, prompt_id, metric, value, rationale))
            
            # Handle llm_judge separately if present
            if "llm_judge" in scores:
                judge_data = scores["llm_judge"]
                if isinstance(judge_data, dict) and not judge_data.get("error"):
                    cursor.execute('''
                        INSERT INTO scored_responses (run_id, model_name, prompt_id, metric_name, value, rationale)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (run_id, model_name, prompt_id, "judge_safety_score", judge_data.get("safety_score"), judge_data.get("rationale", "")))
                    cursor.execute('''
                        INSERT INTO scored_responses (run_id, model_name, prompt_id, metric_name, value, rationale)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (run_id, model_name, prompt_id, "judge_quality_score", judge_data.get("quality_score"), judge_data.get("rationale", "")))
            
            conn.commit()

    def save_aggregated_summary(self, run_id: str, summaries: List[Dict[str, Any]]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for s in summaries:
                cursor.execute('''
                    INSERT INTO aggregated_summary 
                    (run_id, model_name, category, mean_score, safety_score, 
                     mean_latency_ms, median_latency_ms, p95_latency_ms, tokens_per_sec, narrative_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id, s['model_name'], s.get('category', 'OVERALL'), s.get('mean_score'), 
                    s.get('overall_safety_score'), s.get('mean_latency_ms'), s.get('median_latency_ms'),
                    s.get('p95_latency_ms'), s.get('tokens_per_sec'), s.get('narrative_summary')
                ))
            conn.commit()

    def get_latest_run_id(self) -> Optional[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None

    def get_run_results(self, run_id: str) -> List[Dict]:
        """Returns combined raw and scored results for a run."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # This is complex because scores are in a separate table.
            # For now, let's just get the raw responses and then we can join or aggregate.
            cursor.execute('''
                SELECT r.*, group_concat(s.metric_name || ':' || s.value, '|') as scores
                FROM raw_responses r
                LEFT JOIN scored_responses s ON r.run_id = s.run_id AND r.model_name = s.model_name AND r.prompt_id = s.prompt_id
                WHERE r.run_id = ?
                GROUP BY r.id
            ''', (run_id,))
            return [dict(row) for row in cursor.fetchall()]
