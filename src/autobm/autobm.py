import os
import re
import random
import numpy as np
import pickle
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable
import traceback
import datetime

from .client import LLMClient
from .model.pipeline import visualize_test
from .utils import code_impl, direct_construct, create_model_adapter
from .prompt.manager import PromptManager

patterns = {
    "code":       re.compile(r'```python\s*([\s\S]*?)\s*```', flags=re.DOTALL|re.IGNORECASE),
    "policy":     re.compile(r'```python\s*([\s\S]*?)\s*```', flags=re.DOTALL|re.IGNORECASE),
    "conclusion": re.compile(r"Conclusion(.*)",            flags=re.DOTALL),
}

@dataclass
class ModelElement:
    policy:      str
    code:        str
    best_params: Any
    test_loss:   float
    # test_mse:    float # Removed MSE
    judgement:   str
    # vis:         str # Removed vis (from visualize_brief)

def tournament_selection(pool: List[ModelElement],
                         tournament_size: int,
                         sample_size:     int
                        ) -> List[ModelElement]:
    candidates = pool.copy()
    selected = []
    num_to_select = min(sample_size, len(pool))
    for _ in range(num_to_select):
        if not candidates: break
        current_tournament_size = min(tournament_size, len(candidates))
        if current_tournament_size == 0: break
        
        tournament_candidates = random.sample(candidates, current_tournament_size)
        winner = min(tournament_candidates, key=lambda m: m.test_loss)
        selected.append(winner)
        candidates.remove(winner)
    return selected

@dataclass
class AutoBM:
    client:       LLMClient
    pm:           PromptManager
    train:        Any
    val:          Any
    test:         Any
    output_dir:   str = './output'

    pool:         List[ModelElement] = field(default_factory=list)
    all_losses:   List[float]        = field(default_factory=list)
    stats:        List[dict]         = field(default_factory=list)
    
    max_pool_size:    int = 100
    kill_batch:       int = 20 
    init_size:        int = 5

    logs_dir:                       Optional[str] = None
    log_path:                       Optional[str] = None
    intermediate_summary_dir:       Optional[str] = None
    expansion_round_summary_dir:    Optional[str] = None
    final_pkl_dir:                  Optional[str] = None
    final_csv_dir:                  Optional[str] = None

    def __post_init__(self):
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        self.log_path = os.path.join(self.logs_dir, 'autobm_run.log')
        
        self.intermediate_summary_dir = os.path.join(self.logs_dir, 'summaries')
        self.expansion_round_summary_dir = os.path.join(self.intermediate_summary_dir, 'expansion_rounds')
        
        self.final_pkl_dir = os.path.join(self.output_dir, 'pkl')
        self.final_csv_dir = os.path.join(self.output_dir, 'csv')

        for path in [self.output_dir, self.logs_dir, self.intermediate_summary_dir, 
                     self.expansion_round_summary_dir, self.final_pkl_dir, self.final_csv_dir]:
            if path: os.makedirs(path, exist_ok=True)
        
        self._log_message(f"AutoBM process configured. Final outputs: '{self.output_dir}', Logs/Intermediate: '{self.logs_dir}'", new_log=True)

    def _log_message(self, message: str, new_log: bool = False):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        if self.log_path:
            mode = 'w' if new_log else 'a'
            try:
                with open(self.log_path, mode, encoding='utf-8') as lf:
                    lf.write(log_entry + '\n')
            except Exception as e:
                print(f"[{timestamp}] [Critical] Failed to write to log file {self.log_path}: {e}")

    def _cleanup_pool(self):
        if len(self.pool) >= self.max_pool_size:
            original_size = len(self.pool)
            self.pool.sort(key=lambda m: m.test_loss, reverse=True) 
            self.pool = self.pool[self.kill_batch:] 
            self._log_message(f"[Info] Pool cleanup: original size {original_size} (≥{self.max_pool_size}), "
                              f"killed {self.kill_batch}. New pool size: {len(self.pool)}")

    def _get_model(self, policy: str) -> Optional[ModelElement]:
        try:
            code = code_impl(policy=policy, pattern=patterns['code'], client=self.client, pm=self.pm, train_loader=self.train)
            if not code: self._log_message(f"[Warn] Code implementation failed."); return None

            # Get current task to determine adapter type
            from .model.pipeline import PIPELINE_TASK_NAME
            adapter_type = "cda" if PIPELINE_TASK_NAME == "cda" else "batch"
            model_adapter = create_model_adapter(code, adapter_type)

            # Train the model using the adapter
            trained_model_or_params, _ = model_adapter.train(self.train, self.val)
            test_results = model_adapter.test(self.val)

            if not test_results: self._log_message(f"[Warn] Test model returned no results."); return None

            losses = [r['loss'] for r in test_results if 'loss' in r and r['loss'] is not None]
            if not losses: self._log_message(f"[Warn] No valid loss values in test results."); return None

            loss = float(np.mean(losses))
            if np.isnan(loss) or np.isinf(loss): self._log_message(f"[Warn] Model resulted in NaN/Inf loss."); return None

            params = trained_model_or_params if adapter_type == "cda" else trained_model_or_params.state_dict()
            vis_detail = visualize_test(test_results=test_results, params=params)

            judge_p = self.pm.render_prompt('judge.j2', policy=policy, vis=vis_detail) # vis_detail is still used for judgement
            judgment = direct_construct(judge_p, patterns['conclusion'], self.client)

            self._log_message(f"[Info] Model generated. Loss: {loss:.4f}") # MSE removed from log
            # For CDA models, we store the parameters; for batch models, we store the state dict
            model_state = trained_model_or_params if adapter_type == "cda" else trained_model_or_params.state_dict()
            return ModelElement(policy, code, model_state, loss, judgment or "N/A") # MSE and vis removed from constructor

        except Exception as e:
            self._log_message(f"[Warn] Model generation/training failed: {e}\n{traceback.format_exc(limit=2)}")
            return None

    def _update_evolution_stats(self, new_model: ModelElement):
        self.pool.append(new_model)
        self.all_losses.append(new_model.test_loss)

        current_best_loss = min(self.all_losses) if self.all_losses else float('inf')
        current_avg_loss = sum(self.all_losses) / len(self.all_losses) if self.all_losses else float('inf')

        self.stats.append({'new_test_loss': new_model.test_loss, 'best_test_loss': current_best_loss, 'avg_test_loss': current_avg_loss})
        self._log_message(f"[Info] Added model. Pool: {len(self.pool)}, Loss: {new_model.test_loss:.4f}, Best: {current_best_loss:.4f}, Avg: {current_avg_loss:.4f}")
        self._cleanup_pool()

    def _format_simple_policy(self, sampled_models: List[ModelElement]) -> str:
        return "\n".join(f"Policy {i+1}: {e.policy}" for i, e in enumerate(sampled_models))

    def _format_complex_model_details(self, sampled_models: List[ModelElement]) -> str:
        parts = [self.pm.render_prompt('model.j2', 
                                     policy=e.policy, 
                                     # vis=e.vis, # Removed vis
                                     judgement=e.judgement) 
                 for e in sampled_models]
        return "\n---\n".join(parts)

    def _generate_and_evaluate_new_model(self, selection_fn: Callable[[], List[ModelElement]], prompt_template_name: str,
                                         prompt_kwargs: dict, output_pattern: re.Pattern, formatter_fn: Callable[[List[ModelElement]], str]):
        if not self.pool: self._log_message(f"[Warn] Pool empty for {prompt_template_name}."); return

        parent_models = selection_fn()
        if not parent_models: self._log_message(f"[Warn] No parents for {prompt_template_name}. Pool: {len(self.pool)}"); return

        formatted_models_str = formatter_fn(parent_models)
        prompt = self.pm.render_prompt(prompt_template_name, models=formatted_models_str, **prompt_kwargs)
        try:
            new_policy = direct_construct(prompt, output_pattern, self.client)
            if not new_policy: self._log_message(f"[Warn] No policy from direct_construct via {prompt_template_name}."); return
        except Exception as e: self._log_message(f"[Warn] direct_construct failed for {prompt_template_name}: {e}"); return
            
        new_model_element = self._get_model(new_policy)
        if new_model_element: self._update_evolution_stats(new_model_element)
        else: self._log_message(f"[Warn] Invalid model from policy via {prompt_template_name}")

    def init_population(self, n: Optional[int] = None):
        num_to_init = n or self.init_size
        self._log_message(f"[Info] Initializing population: {num_to_init} models.")
        base_prompt = self.pm.render_prompt('init.j2')
        generated_count = 0
        for i in range(num_to_init):
            self._log_message(f"[Info] Generating initial model {i+1}/{num_to_init}")
            try:
                policy = direct_construct(base_prompt, patterns['policy'], self.client)
                if not policy: self._log_message(f"[Warn] Initial policy {i+1} gen failed."); continue
                model_element = self._get_model(policy)
                if model_element: self._update_evolution_stats(model_element); generated_count += 1
            except Exception as e: self._log_message(f"[Warn] Initial model {i+1} gen error: {e}\n{traceback.format_exc(limit=1)}")
        
        self._log_message(f"[Info] Initial population generated: {generated_count}/{num_to_init} successful.")
        if self.intermediate_summary_dir:
            self.save_state_summary(os.path.join(self.intermediate_summary_dir, '00_init_population_state.txt'))

    def expansion(self, rounds: int = 20, improve_size_per_round: int = 3, explore_size_per_round: int = 2,
                  selection_sample_size: int = 3, tournament_k_size: int = 5):
        self._log_message(f"[Info] Starting expansion: {rounds} rounds.")
        for r_idx in range(rounds):
            current_round = r_idx + 1
            self._log_message(f"[Info] Expansion round {current_round}/{rounds}. Pool: {len(self.pool)}")
            
            if not self.pool:
                self._log_message("[Warn] Pool empty at start of round.")
                if self.expansion_round_summary_dir:
                    self.save_state_summary(os.path.join(self.expansion_round_summary_dir, f'round_{current_round:02d}_pool_empty.txt'))
                continue

            for i in range(explore_size_per_round): 
                self._log_message(f"[Info] R{current_round}-Explore {i+1}/{explore_size_per_round}")
                if not self.pool: self._log_message("[Warn] Pool empty during explore."); break
                self._generate_and_evaluate_new_model(lambda: random.sample(self.pool, min(selection_sample_size, len(self.pool))),
                                                     'differ.j2', {}, patterns['policy'], self._format_simple_policy)
            
            for i in range(improve_size_per_round): 
                self._log_message(f"[Info] R{current_round}-Improve {i+1}/{improve_size_per_round}")
                if not self.pool: self._log_message("[Warn] Pool empty during improve."); break
                self._generate_and_evaluate_new_model(lambda: tournament_selection(self.pool, tournament_k_size, selection_sample_size),
                                                     'optimistic.j2', {}, patterns['policy'], self._format_complex_model_details)
            
            if self.expansion_round_summary_dir:
                self.save_state_summary(os.path.join(self.expansion_round_summary_dir, f'round_{current_round:02d}_state.txt'))
            self._log_message(f"[Info] End of round {current_round}. Pool: {len(self.pool)}.")

    def save_state_summary(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"# AutoBM Pool State Summary\n")
                f.write(f"# Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Models in pool: {len(self.pool)}\n")
                if self.all_losses:
                    f.write(f"# Best loss: {min(self.all_losses):.4f}, Avg loss: {(sum(self.all_losses)/len(self.all_losses)):.4f}\n")
                else: f.write(f"# No models/losses recorded.\n")
                f.write("\n---\n")
                
                sorted_pool = sorted(self.pool, key=lambda x: x.test_loss)
                for i, m in enumerate(sorted_pool, 1):
                    f.write(f"## Rank {i}: Loss={m.test_loss:.4f}\n") # MSE removed
                    # f.write(f" Vis: {m.vis}\n Judgement: {m.judgement}\n") # Vis removed
                    f.write(f" Judgement: {m.judgement}\n")
                    f.write(f" Policy:\n```python\n{m.policy}\n```\n")
                    f.write(f" Code:\n```python\n{m.code}\n```\n---\n")
            self._log_message(f"[Info] State summary saved: {path}")
        except Exception as e: self._log_message(f"[Error] Saving state summary to {path} failed: {e}")

    def dump_full_state(self):
        self._log_message(f"[Info] Dumping final state to {self.output_dir}")

        if self.pool and self.final_pkl_dir:
            try:
                with open(os.path.join(self.final_pkl_dir, 'pool.pkl'), 'wb') as f: pickle.dump(self.pool, f)
                self._log_message(f"[Info] Saved pool.pkl: {len(self.pool)} models.")
            except Exception as e: self._log_message(f"[Error] Saving pool.pkl failed: {e}")
        elif not self.final_pkl_dir: self._log_message("[Warn] Final PKL dir not set. Skipping pool.pkl.")
        else: self._log_message("[Info] Pool empty. Skipping pool.pkl.")

        if self.stats and self.final_csv_dir:
            try:
                with open(os.path.join(self.final_csv_dir, 'model_stats.csv'), 'w', newline='', encoding='utf-8') as cf:
                    fieldnames = list(self.stats[0].keys()) if self.stats and isinstance(self.stats[0], dict) else []
                    if fieldnames:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        writer.writeheader(); writer.writerows(self.stats)
                        self._log_message(f"[Info] Saved model_stats.csv: {len(self.stats)} entries.")
                    else: self._log_message("[Info] Stats empty or format issue. Skipping model_stats.csv.")
            except Exception as e: self._log_message(f"[Error] Saving model_stats.csv failed: {e}")
        elif not self.final_csv_dir: self._log_message("[Warn] Final CSV dir not set. Skipping model_stats.csv.")
        else: self._log_message("[Info] Stats empty. Skipping model_stats.csv.")

    def clear_state(self):
        self._log_message("[Info] Clearing internal state.")
        self.pool.clear(); self.all_losses.clear(); self.stats.clear()

    def start(self, initial_population_size: int = 5, expansion_rounds: int = 20, # Method name changed back to start
                            improve_steps_per_round: int = 5, explore_steps_per_round: int = 5,
                            model_selection_sample_size: int = 3, tournament_k_size: int = 5):
        self._log_message(f"[Info] ===== Starting New AutoBM Run =====")
        self.clear_state()
        try:
            self.init_population(n=initial_population_size)
            if not self.pool: self._log_message("[Warning] Initial population empty after init phase.")
            
            self.expansion(rounds=expansion_rounds, improve_size_per_round=improve_steps_per_round,
                           explore_size_per_round=explore_steps_per_round, selection_sample_size=model_selection_sample_size,
                           tournament_k_size=tournament_k_size)
            
            if not self.all_losses: self._log_message("[Error] Run completed, but no models/losses recorded.")
            else: self._log_message(f"[Info] Run completed! Best loss: {min(self.all_losses):.4f}")
            
        except Exception as e:
            self._log_message(f"[Critical] Unhandled error during run: {e}\n{traceback.format_exc()}")
        finally:
            self._log_message("[Info] Attempting to dump final state...")
            self.dump_full_state()
            self._log_message("[Info] ===== AutoBM Run Finished =====")