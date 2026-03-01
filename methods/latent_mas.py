import json
import re
import copy
from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_message_sequential_latent_mas,
    build_agent_message_hierarchical_latent_mas,
    build_meta_agent_message_hierarchical_latent_mas,
)
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
import pdb

try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = None
    DynamicCache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.greedy = bool(getattr(args, "greedy", False)) if args else False
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        if getattr(args, "use_vllm", False):
            from vllm import SamplingParams
            sampling_temperature = temperature if not self.greedy else 0.0
            sampling_top_p = top_p if not self.greedy else 1.0
            self.sampling_params = SamplingParams(
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                max_tokens=args.max_new_tokens,
            )
        else:
            self.sampling_params = None
        self.task = args.task
        self._choice_token_id_cache: Optional[Dict[str, int]] = None

    def _get_probe_model_and_device(self):
        if hasattr(self.model, "model"):
            return self.model.model, self.model.device
        if hasattr(self.model, "HF_model"):
            return self.model.HF_model, self.model.HF_device
        return None, None

    def _get_choice_token_ids(self) -> Dict[str, int]:
        if self._choice_token_id_cache is not None:
            return self._choice_token_id_cache

        tokenizer = self.model.tokenizer
        token_ids: Dict[str, int] = {}
        for label in ["A", "B", "C", "D"]:
            chosen_id: Optional[int] = None
            for variant in [label, f" {label}", label.lower(), f" {label.lower()}"]:
                ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    chosen_id = int(ids[0])
                    break
            if chosen_id is None:
                fallback_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
                if not fallback_ids:
                    raise ValueError(f"Unable to map choice token id for label '{label}'")
                chosen_id = int(fallback_ids[0])
            token_ids[label.lower()] = chosen_id

        self._choice_token_id_cache = token_ids
        return token_ids

    @torch.no_grad()
    def _score_choice_logits_from_cache(
        self,
        past_kv: Optional[Tuple],
        past_kv_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> List[Dict[str, Dict[str, float]]]:
        if past_kv is None:
            return [
                {
                    "choice_logits": {"a": float("nan"), "b": float("nan"), "c": float("nan"), "d": float("nan")},
                    "choice_probs": {"a": float("nan"), "b": float("nan"), "c": float("nan"), "d": float("nan")},
                }
                for _ in range(batch_size)
            ]

        probe_model, probe_device = self._get_probe_model_and_device()
        if probe_model is None:
            return [
                {
                    "choice_logits": {"a": float("nan"), "b": float("nan"), "c": float("nan"), "d": float("nan")},
                    "choice_probs": {"a": float("nan"), "b": float("nan"), "c": float("nan"), "d": float("nan")},
                }
                for _ in range(batch_size)
            ]

        probe_text = "\nAnswer with exactly one letter: A, B, C, or D.\nAnswer: "
        probe_batch = [probe_text] * batch_size
        probe_encoded = self.model.tokenizer(
            probe_batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        probe_ids = probe_encoded["input_ids"].to(probe_device)
        probe_mask = probe_encoded["attention_mask"].to(probe_device)

        past_len = _past_length(past_kv)
        if past_len > 0:
            # Use provided past mask (0/1 for left-padding) so we don't attend to padding.
            if past_kv_mask is not None and past_kv_mask.shape == (batch_size, past_len):
                prefix_mask = past_kv_mask.to(device=probe_mask.device, dtype=probe_mask.dtype)
            else:
                prefix_mask = torch.ones(
                    (batch_size, past_len),
                    dtype=probe_mask.dtype,
                    device=probe_mask.device,
                )
            full_mask = torch.cat([prefix_mask, probe_mask], dim=-1)
        else:
            full_mask = probe_mask

        # Never probe with the live cache: clone so probe forward or cache APIs don't mutate it.
        try:
            past_for_probe = copy.deepcopy(past_kv)
        except (AttributeError, TypeError, NotImplementedError):
            # Some cache types don't deepcopy; truncate returns new tuple for legacy caches (DynamicCache mutates in place).
            past_for_probe = self._truncate_past(past_kv, past_len)

        outputs = probe_model(
            input_ids=probe_ids,
            attention_mask=full_mask,
            past_key_values=past_for_probe,
            use_cache=False,
            return_dict=True,
        )
        next_token_logits = outputs.logits[:, -1, :]

        choice_ids = self._get_choice_token_ids()
        ordered = ["a", "b", "c", "d"]
        choice_logits_tensor = torch.stack(
            [next_token_logits[:, choice_ids[label]] for label in ordered],
            dim=-1,
        )
        choice_probs_tensor = torch.softmax(choice_logits_tensor, dim=-1)

        records: List[Dict[str, Dict[str, float]]] = []
        for row_idx in range(batch_size):
            logits_row = choice_logits_tensor[row_idx]
            probs_row = choice_probs_tensor[row_idx]
            records.append(
                {
                    "choice_logits": {
                        label: float(logits_row[col_idx].item())
                        for col_idx, label in enumerate(ordered)
                    },
                    "choice_probs": {
                        label: float(probs_row[col_idx].item())
                        for col_idx, label in enumerate(ordered)
                    },
                }
            )
        return records

    @staticmethod
    def _default_meta_prompts(worker_roles: List[str]) -> Dict[str, str]:
        defaults = [
            "Use a decomposition-first strategy: break the task into explicit subproblems and solve step by step.",
            "Use a verification-first strategy: challenge assumptions, test alternatives, and justify each step.",
            "Use an efficiency-first strategy: target a concise solution path with minimal but sufficient reasoning.",
        ]
        role_to_prompt: Dict[str, str] = {}
        for idx, role in enumerate(worker_roles):
            role_to_prompt[role] = defaults[min(idx, len(defaults) - 1)]
        return role_to_prompt

    @staticmethod
    def _normalize_meta_value(value) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return ""

    @staticmethod
    def _parse_meta_json(raw_text: str) -> List[str]:
        candidates: List[str] = []
        stripped = raw_text.strip()
        if stripped:
            candidates.append(stripped)

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())

        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidates.append(raw_text[first_brace : last_brace + 1].strip())

        for candidate in candidates:
            try:
                loaded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(loaded, dict):
                continue
            if isinstance(loaded.get("worker_prompts"), list):
                prompts: List[str] = []
                for item in loaded["worker_prompts"]:
                    normalized = LatentMASMethod._normalize_meta_value(item)
                    if normalized:
                        prompts.append(normalized)
                if prompts:
                    return prompts[:3]

            ordered_worker_keys = ["worker_1", "worker_2", "worker_3", "agent_1", "agent_2", "agent_3"]
            keyed_prompts: List[str] = []
            for key in ordered_worker_keys:
                normalized = LatentMASMethod._normalize_meta_value(loaded.get(key))
                if normalized:
                    keyed_prompts.append(normalized)
            if keyed_prompts:
                return keyed_prompts[:3]

            fallback_roles = ["planner", "critic", "refiner"]
            fallback_prompts: List[str] = []
            for role in fallback_roles:
                normalized = LatentMASMethod._normalize_meta_value(loaded.get(role))
                if normalized:
                    fallback_prompts.append(normalized)
            if fallback_prompts:
                return fallback_prompts[:3]
        return []

    def _generate_hierarchical_meta_prompts(
        self, items: List[Dict]
    ) -> Tuple[List[Dict[str, str]], List[Dict]]:
        worker_roles = [agent.role for agent in self.agents if agent.role != "judger"]
        default_map = self._default_meta_prompts(worker_roles)

        if self.args.prompt != "hierarchical":
            return [{} for _ in items], [{} for _ in items]
        if not bool(getattr(self.args, "use_meta_prompt_generator", False)):
            return [{} for _ in items], [{} for _ in items]

        batch_messages = [
            build_meta_agent_message_hierarchical_latent_mas(question=item["question"], args=self.args)
            for item in items
        ]
        # Meta generation: no_think (enable_thinking=False) and max_tokens=512
        prepare_kwargs = {"enable_thinking": False}
        try:
            prompts, input_ids, attention_mask, _ = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True, **prepare_kwargs
            )
        except TypeError:
            # Tokenizer does not support enable_thinking (e.g. non-Qwen3)
            prompts, input_ids, attention_mask, _ = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
        meta_max_tokens = 512
        if self.model.use_vllm:
            generated = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=meta_max_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
        else:
            generated, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=meta_max_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )

        role_prompts: List[Dict[str, str]] = []
        meta_debug: List[Dict] = []
        for text in generated:
            parsed = self._parse_meta_json(text)
            combined = default_map.copy()
            used_default_roles: List[str] = []
            for idx, role in enumerate(worker_roles):
                if idx < len(parsed) and parsed[idx]:
                    combined[role] = parsed[idx]
                else:
                    used_default_roles.append(role)
            role_prompts.append(combined)
            meta_debug.append(
                {
                    "raw_output": text,
                    "parsed_worker_prompts": parsed,
                    "used_default_roles": used_default_roles,
                }
            )
        return role_prompts, meta_debug

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            # transformers 5.x: to_legacy_cache/from_legacy_cache removed; manually truncate layers
            for layer in past_kv.layers:
                if layer.get_seq_length() <= 0:
                    continue
                keep = min(tokens_to_keep, layer.get_seq_length())
                start = layer.get_seq_length() - keep
                layer.keys = layer.keys[..., start:, :].contiguous()
                layer.values = layer.values[..., start:, :].contiguous()
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length = keep
            return past_kv
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @staticmethod
    def _to_legacy_cache(past_kv):
        if past_kv is None:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            # transformers 5.x: to_legacy_cache removed; extract (k, v) from each layer
            return tuple((layer.keys, layer.values) for layer in past_kv.layers)
        return past_kv

    @staticmethod
    def _from_legacy_cache(reference, legacy_cache):
        if Cache is not None and isinstance(reference, Cache) and DynamicCache is not None:
            # transformers 5.x: from_legacy_cache removed; use DynamicCache(ddp_cache_data=...)
            return DynamicCache(ddp_cache_data=legacy_cache)
        return tuple(legacy_cache)

    def _concat_past_key_values(self, caches: List[Optional[Tuple]]) -> Optional[Tuple]:
        valid_caches = [cache for cache in caches if cache is not None and _past_length(cache) > 0]
        if not valid_caches:
            return None

        legacy_caches = [self._to_legacy_cache(cache) for cache in valid_caches]
        num_layers = len(legacy_caches[0])
        concatenated_layers = []
        for layer_idx in range(num_layers):
            layer_group = [legacy[layer_idx] for legacy in legacy_caches]
            first_layer = layer_group[0]
            if isinstance(first_layer, tuple):
                tensor_count = len(first_layer)
                concatenated_layers.append(
                    tuple(
                        torch.cat([layer[t_idx] for layer in layer_group], dim=-2)
                        for t_idx in range(tensor_count)
                    )
                )
            elif torch.is_tensor(first_layer):
                concatenated_layers.append(torch.cat(layer_group, dim=-2))
            else:
                raise TypeError("Unsupported cache layer type during hierarchical concatenation.")

        concatenated_legacy = tuple(concatenated_layers)
        return self._from_legacy_cache(valid_caches[0], concatenated_legacy)

    def _concat_past_attention_masks(
        self,
        caches: List[Optional[Tuple]],
        masks: List[Optional[torch.Tensor]],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Concatenate per-worker past masks (same order as caches). Each mask must have shape (batch_size, _past_length(cache))."""
        segments: List[torch.Tensor] = []
        for cache, mask in zip(caches, masks):
            if cache is None:
                continue
            cache_len = _past_length(cache)
            if cache_len <= 0:
                continue
            if mask is None:
                segments.append(torch.ones((batch_size, cache_len), dtype=dtype, device=device))
            else:
                if mask.shape != (batch_size, cache_len):
                    raise ValueError(
                        f"worker mask shape must be (batch_size={batch_size}, cache_len={cache_len}), got {mask.shape}"
                    )
                segments.append(mask.to(device=device, dtype=dtype))
        if not segments:
            return None
        return torch.cat(segments, dim=-1)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        hierarchical_mode = self.args.prompt == "hierarchical"
        past_kv: Optional[Tuple] = None
        past_kv_mask: Optional[torch.Tensor] = None
        hierarchical_worker_caches: List[Optional[Tuple]] = []
        hierarchical_worker_masks: List[Optional[torch.Tensor]] = []
        hierarchical_position_offset = torch.zeros(
            batch_size, dtype=torch.long, device=self.model.device
        )
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        meta_role_prompts, meta_debug = self._generate_hierarchical_meta_prompts(items)

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                        meta_prompt=(
                            meta_role_prompts[idx].get(agent.role, "")
                            if agent.role != "judger"
                            else ""
                        ),
                    )
                    for idx, item in enumerate(items)
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                base_past_kv = None if hierarchical_mode else past_kv
                base_past_kv_mask = None if hierarchical_mode else past_kv_mask
                prev_past_len = _past_length(base_past_kv)
                worker_position_offset = None
                if hierarchical_mode and not self.latent_only:
                    worker_position_offset = hierarchical_position_offset.clone()

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                worker_past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=base_past_kv,
                    past_attention_mask=base_past_kv_mask,
                    position_offset=worker_position_offset,
                )
                # Build past mask in lockstep with cache: same inputs (base + wrapped + latent).
                if base_past_kv_mask is None:
                    worker_past_kv_mask = wrapped_mask
                else:
                    worker_past_kv_mask = torch.cat([base_past_kv_mask, wrapped_mask], dim=-1)
                if self.latent_steps > 0:
                    latent_append = torch.ones(
                        (batch_size, self.latent_steps),
                        dtype=worker_past_kv_mask.dtype,
                        device=worker_past_kv_mask.device,
                    )
                    worker_past_kv_mask = torch.cat([worker_past_kv_mask, latent_append], dim=-1)

                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(worker_past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    worker_past_kv = self._truncate_past(worker_past_kv, tokens_to_keep)
                    if tokens_to_keep <= 0:
                        worker_past_kv_mask = None
                    else:
                        worker_past_kv_mask = worker_past_kv_mask[:, -tokens_to_keep:].contiguous()

                if hierarchical_mode:
                    if worker_position_offset is not None:
                        if worker_past_kv_mask is not None:
                            worker_token_counts = worker_past_kv_mask.sum(dim=-1).to(
                                device=hierarchical_position_offset.device,
                                dtype=hierarchical_position_offset.dtype,
                            )
                        else:
                            worker_len = _past_length(worker_past_kv)
                            worker_token_counts = torch.full(
                                (batch_size,),
                                worker_len,
                                device=hierarchical_position_offset.device,
                                dtype=hierarchical_position_offset.dtype,
                            )
                        hierarchical_position_offset = hierarchical_position_offset + worker_token_counts
                    hierarchical_worker_caches.append(worker_past_kv)
                    hierarchical_worker_masks.append(worker_past_kv_mask)
                else:
                    past_kv = worker_past_kv
                    past_kv_mask = worker_past_kv_mask

                verifier_scores = self._score_choice_logits_from_cache(
                    past_kv=worker_past_kv,
                    past_kv_mask=worker_past_kv_mask,
                    batch_size=batch_size,
                )

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "meta_prompt": meta_role_prompts[idx].get(agent.role, "") if self.args.prompt == "hierarchical" and agent.role != "judger" else "",
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "choice_logits": verifier_scores[idx]["choice_logits"],
                            "choice_probs": verifier_scores[idx]["choice_probs"],
                            "output": "",
                        }
                    )
            else:
                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                if hierarchical_mode:
                    past_for_decoding = self._concat_past_key_values(hierarchical_worker_caches)
                    past_for_decoding_mask = self._concat_past_attention_masks(
                        hierarchical_worker_caches,
                        hierarchical_worker_masks,
                        batch_size=batch_size,
                        dtype=judger_mask.dtype,
                        device=judger_mask.device,
                    )
                else:
                    past_for_decoding = past_kv
                    past_for_decoding_mask = past_kv_mask
                if self.latent_steps <= 0:
                    past_for_decoding = None
                    past_for_decoding_mask = None

                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=not self.greedy,
                    past_key_values=past_for_decoding,
                    past_attention_mask=past_for_decoding_mask,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "meta_agent": meta_debug[idx] if idx < len(meta_debug) else {},
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        hierarchical_mode = self.args.prompt == "hierarchical"
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        meta_role_prompts, meta_debug = self._generate_hierarchical_meta_prompts(items)
        hierarchical_position_offset = torch.zeros(
            batch_size, dtype=torch.long, device=self.model.HF_device
        )

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                        meta_prompt=(
                            meta_role_prompts[idx].get(agent.role, "")
                            if agent.role != "judger"
                            else ""
                        ),
                    )
                    for idx, item in enumerate(items)
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                base_past_kv = None if hierarchical_mode else past_kv
                prev_past_len = _past_length(base_past_kv)
                worker_position_offset = None
                if hierarchical_mode and not self.latent_only:
                    worker_position_offset = hierarchical_position_offset.clone()

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                worker_past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=base_past_kv,
                    position_offset=worker_position_offset,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(worker_past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    worker_past_kv = self._truncate_past(worker_past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if (self.sequential_info_only or self.latent_only) and not hierarchical_mode:
                    embedding_record = embedding_record[-1:]

                if not hierarchical_mode:
                    past_kv = worker_past_kv
                elif worker_position_offset is not None:
                    worker_token_counts = wrapped_mask.sum(dim=-1).to(
                        device=hierarchical_position_offset.device,
                        dtype=hierarchical_position_offset.dtype,
                    ) + int(self.latent_steps)
                    if self.sequential_info_only:
                        if self.latent_only:
                            keep_counts = torch.full_like(worker_token_counts, int(self.latent_steps))
                        else:
                            worker_len = _past_length(worker_past_kv)
                            keep_counts = torch.full_like(worker_token_counts, int(worker_len))
                        worker_token_counts = torch.minimum(worker_token_counts, keep_counts)
                    hierarchical_position_offset = hierarchical_position_offset + worker_token_counts

                verifier_scores = self._score_choice_logits_from_cache(
                    past_kv=worker_past_kv,
                    past_kv_mask=None,
                    batch_size=batch_size,
                )
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "meta_prompt": meta_role_prompts[idx].get(agent.role, "") if self.args.prompt == "hierarchical" and agent.role != "judger" else "",
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "choice_logits": verifier_scores[idx]["choice_logits"],
                            "choice_probs": verifier_scores[idx]["choice_probs"],
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "meta_agent": meta_debug[idx] if idx < len(meta_debug) else {},
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
