from __future__ import annotations
import os
import time
import json
import traceback
import threading
import unicodedata
import re
from typing import List, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

from config import (
    LLM_SLOTS, LLM_GPU_LAYERS,
    LLM_MAX_NEW_TOKENS, LLM_TEMP, LLM_TOP_P,
    LLM_CONTEXT_LEN, LLM_STOP_SEQUENCES, LLM_EOS_STRINGS,
    LLM_MIN_OUTPUT_CHARS, LLM_REPETITION_PENALTY, LLM_TOP_K,
)

try:
    from config import LLM_THREADS
except Exception:
    LLM_THREADS = 0  # 0 => auto (os.cpu_count()-1)
try:
    from config import LLM_ENABLE_RETRY
except Exception:
    LLM_ENABLE_RETRY = False
try:
    from config import LLM_AUTOSIZE_MAX_NEW_TOKENS
except Exception:
    LLM_AUTOSIZE_MAX_NEW_TOKENS = True
try:
    from config import LLM_WARMUP
except Exception:
    LLM_WARMUP = True
try:
    from config import LLM_TUNING_CACHE_MODE
except Exception:
    LLM_TUNING_CACHE_MODE = "off"  # 'off'|'hint'|'on'

LLM_AUTBOOST_ENABLE = True


_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_NEWLINES_RE = re.compile(r"\r\n?")

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _NEWLINES_RE.sub("\n", s)
    s = _MULTISPACE_RE.sub(" ", s)
    return s.strip()

def _apply_stop_sequences(text: str, stops: Sequence[str]) -> str:
    if not text or not stops:
        return text
    cut_idx = None
    for st in stops:
        if not st:
            continue
        i = text.find(st)
        if i != -1:
            cut_idx = i if cut_idx is None else min(cut_idx, i)
    return text[:cut_idx] if cut_idx is not None else text

def _looks_like_cut_word(text: str) -> bool:
    if not text:
        return False
    s = text.rstrip()
    if not s:
        return False
    last = s[-1]
    if last in ".!?…":
        return False
    if last.isalnum():
        return True
    if last in ",;:-—":
        return True
    return False


def _post_process(text: str, stops: Sequence[str]) -> str:
    text = _apply_stop_sequences(text, stops)
    text = _normalize_text(text)
    if text and text[-1] not in ".!?…" and not _looks_like_cut_word(text):
        text += "."
    return text


class LLMManager:
    def __init__(self):
        self.backend: Optional[str] = None
        self.model = None
        self._gen_lock = threading.Lock()

        self._ctx_len = int(LLM_CONTEXT_LEN) if LLM_CONTEXT_LEN else 4096
        self._stops: List[str] = list(LLM_STOP_SEQUENCES or [])
        self._eos_strings: List[str] = list(LLM_EOS_STRINGS or [])
        self._min_chars = int(LLM_MIN_OUTPUT_CHARS or 220)

        # bazowe wątki CPU
        if int(LLM_THREADS or 0) > 0:
            self._base_threads = int(LLM_THREADS)
        else:
            self._base_threads = max(1, (os.cpu_count() or 2) - 1)
        self._threads = self._base_threads

        # slot/model
        slot = LLM_SLOTS.get("bielik") or {}
        self.repo = slot.get("REPO") or ""
        self.filename = slot.get("FILENAME") or ""
        self.model_type_hint = (slot.get("MODEL_TYPE") or "").strip().lower() or None

        # stan tuningu
        self._supports_n_batch = False
        self._supports_batch_size = False
        self._prefill_batch = 0

        # efektywne warstwy w TEJ sesji
        self._gpu_layers_effective = int(LLM_GPU_LAYERS or 0)

        self._persist_layers_cap = int(LLM_GPU_LAYERS or 0)  # będzie nadpisany z cache, jeśli istnieje

        # ścieżka do pliku z wynikiem autotuningu
        self._tuning_path = os.path.abspath(".llm_tuning.json")

    def _load_ctransformers(self, repo: str, filename: str, gpu_layers: int):
        from ctransformers import AutoModelForCausalLM
        print(f"[LLM] ctransformers: repo={repo} file={filename} "
              f"(gpu_layers={gpu_layers}, ctx={self._ctx_len}, threads={self._threads})")

        base_kwargs = dict(
            model_file=filename,
            gpu_layers=int(gpu_layers),
            context_length=self._ctx_len,
        )

        # 1) auto-detect
        try:
            return AutoModelForCausalLM.from_pretrained(repo, **base_kwargs)
        except TypeError:
            pass
        except Exception as e:
            msg = "".join(traceback.format_exception_only(type(e), e)).strip()
            print("[LLM][INFO] auto-detect failed, retry with explicit model_type. Reason:", msg)

        # 2) kandydaci
        candidates = []
        if self.model_type_hint:
            candidates.append(self.model_type_hint)
        candidates += ["llama", "mpt", "falcon", "gpt_neox", "gptj"]

        last_err = None
        for mt in candidates:
            try:
                print(f"[LLM] retry with model_type={mt}")
                return AutoModelForCausalLM.from_pretrained(repo, model_type=mt, **base_kwargs)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"ctransformers could not load the model (tried types: {candidates}). Last error: {last_err}")

    def _try_load_with_layers(self, layers: int) -> bool:
        try:
            self.model = self._load_ctransformers(self.repo, self.filename, int(layers))
            self.backend = "ctransformers"
            self._gpu_layers_effective = int(layers)
            print(f"[LLM] Załadowano (GPU={layers}, ctransformers): {self.filename}")
            return True
        except Exception as e:
            print(f"[LLM][WARN] load fail for gpu_layers={layers}:",
                  "".join(traceback.format_exception_only(type(e), e)).strip())
            return False

    def _load_with_best_layers(self, hint_layers: int) -> bool:
        cand = []
        start = max(0, int(hint_layers))
        if start > 0:
            cand.append(start)
            x = start
            while x > 16:
                x = max(16, x - 16)
                if x not in cand:
                    cand.append(x)
            for step in (8, 4, 2, 1):
                if step not in cand and step < cand[-1]:
                    cand.append(step)
        cand.append(0)

        tried = set()
        for c in cand:
            if c in tried:
                continue
            tried.add(c)
            if c == 0:
                try:
                    self.model = self._load_ctransformers(self.repo, self.filename, 0)
                    self.backend = "ctransformers"
                    self._gpu_layers_effective = 0
                    print(f"[LLM] Załadowano (CPU, ctransformers): {self.filename}")
                    return True
                except Exception as e2:
                    print("[LLM][ERROR] ctransformers (CPU) nieudane:",
                          "".join(traceback.format_exception_only(type(e2), e2)).strip())
                    return False
            if self._try_load_with_layers(c):
                return True
        return False

    def _gpu_layers_autoboost_session(self, upper_cap: int):
        if not LLM_AUTBOOST_ENABLE:
            return
        base = int(self._gpu_layers_effective or 0)
        if base <= 0 or self.backend != "ctransformers":
            return
        if upper_cap <= 0:
            return

        last_ok = min(base, upper_cap)
        if base > upper_cap:
            self._gpu_layers_effective = last_ok
            return

        for step in (14, 16):
            cand = last_ok + step
            while cand <= upper_cap:
                print(f"[LLM] gpu_layers autoboost probe → {cand} (cap={upper_cap})")
                try:
                    new_model = self._load_ctransformers(self.repo, self.filename, cand)
                    old_model = self.model
                    self.model = new_model
                    self._gpu_layers_effective = cand
                    print(f"[LLM] gpu_layers ↑ {last_ok} → {cand} (OK, session only)")
                    last_ok = cand
                    try:
                        del old_model
                    except Exception:
                        pass
                    cand += step
                except Exception as e:
                    print(f"[LLM] gpu_layers autoboost fail ({cand}), stay at {last_ok}:",
                          "".join(traceback.format_exception_only(type(e), e)).strip())
                    return

    def _probe_generation_kwargs(self):
        if int(self._gpu_layers_effective or 0) <= 0:
            return

        key = None
        try:
            _ = self.model("ok", max_new_tokens=1, temperature=0.0, top_p=0.0,
                           threads=self._threads, stop=[], n_batch=64)
            self._supports_n_batch = True
            key = "n_batch"
        except TypeError:
            pass
        except Exception:
            pass
        if key is None:
            try:
                _ = self.model("ok", max_new_tokens=1, temperature=0.0, top_p=0.0,
                               threads=self._threads, stop=[], batch_size=64)
                self._supports_batch_size = True
                key = "batch_size"
            except TypeError:
                pass
            except Exception:
                pass
        if key is None:
            return

        for val in (128, 192, 256, 320, 384, 448, 512):
            try:
                kwargs = {key: val}
                _ = self.model("OK.", max_new_tokens=1, temperature=0.0, top_p=0.0,
                               threads=self._threads, stop=[], **kwargs)
                self._prefill_batch = val
                print(f"[LLM] probe: using {key}={val}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
            except Exception:
                break

        if key == "n_batch":
            self._supports_n_batch = self._prefill_batch > 0
        else:
            self._supports_batch_size = self._prefill_batch > 0

    def _measure_tps(self, threads: int) -> float:
        try:
            t0 = time.perf_counter()
            _ = self.model("OK.", max_new_tokens=24, temperature=0.0, top_p=0.0,
                           threads=int(threads), stop=[])
            dt = max(1e-6, time.perf_counter() - t0)
            return 24.0 / dt
        except Exception:
            return 0.0

    def _choose_threads_gpu_biased(self):
        if int(LLM_THREADS or 0) > 0:
            print(f"[LLM] threads pinned by config: {LLM_THREADS}")
            self._threads = int(LLM_THREADS)
            return

        cand = [self._base_threads]
        for v in (8, 6, 4, 2, 1):
            if self._base_threads > v:
                cand.append(v)
        cand = sorted(set(cand), reverse=True)

        scores = {}
        for th in cand:
            tps = self._measure_tps(th)
            scores[th] = tps
            print(f"[LLM] probe: threads={th} → {tps:.1f} tok/s")

        best_th = max(scores, key=scores.get)
        best_tps = scores[best_th]

        pick = best_th
        for th in sorted(scores):  # preferuj mniejsze CPU przy ≥80% best
            if scores[th] >= 0.80 * best_tps:
                pick = th
                break

        self._threads = int(pick)
        print(f"[LLM] threads set → {self._threads} (best={best_th}@{best_tps:.1f} tok/s, gpu-biased pick)")

    def _save_tuning(self):
        if LLM_TUNING_CACHE_MODE != "on":
            return
        try:
            cap_before = int(self._persist_layers_cap or 0)
            cap_after = int(min(self._persist_layers_cap or 1e9, self._gpu_layers_effective))
            self._persist_layers_cap = cap_after  # aktualizuj w pamięci

            data = {
                "model_file": self.filename,
                "persist_layers": cap_after,
                "session_layers_last": int(self._gpu_layers_effective),
                "prefill_batch": int(self._prefill_batch),
                "supports_n_batch": bool(self._supports_n_batch),
                "supports_batch_size": bool(self._supports_batch_size),
                "threads": int(self._threads),
            }
            with open(self._tuning_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[LLM] tuning saved: persist_layers {cap_before} -> {cap_after} | session_layers_last={self._gpu_layers_effective}")
        except Exception:
            pass

    def _load_tuning(self) -> bool:
        if LLM_TUNING_CACHE_MODE not in ("hint", "on"):
            return False
        try:
            if not os.path.isfile(self._tuning_path):
                return False
            with open(self._tuning_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model_file") != self.filename:
                return False

            persist = data.get("persist_layers")
            if persist is None:
                persist = data.get("gpu_layers_effective")

            if persist is not None:
                self._persist_layers_cap = int(persist)
            self._prefill_batch = int(data.get("prefill_batch", 0))
            self._supports_n_batch = bool(data.get("supports_n_batch", False))
            self._supports_batch_size = bool(data.get("supports_batch_size", False))
            self._threads = int(data.get("threads", self._threads))

            print(f"[LLM] tuning restored → persist_layers={self._persist_layers_cap}, "
                  f"last_session_layers={data.get('session_layers_last')}")
            return True
        except Exception:
            return False

    def load(self):
        if not (self.repo and self.filename):
            raise RuntimeError("Brak REPO/FILENAME dla slotu 'bielik'.")

        restored = self._load_tuning()

        cfg_cap = int(LLM_GPU_LAYERS or 0)
        if cfg_cap > 0:
            if self._persist_layers_cap > 0:
                self._persist_layers_cap = min(self._persist_layers_cap, cfg_cap)
            else:
                self._persist_layers_cap = cfg_cap

        if self._persist_layers_cap > 0 and cfg_cap > 0:
            hint_layers = min(self._persist_layers_cap, cfg_cap)
        elif self._persist_layers_cap > 0:
            hint_layers = self._persist_layers_cap
        else:
            hint_layers = cfg_cap  # może być 0 -> CPU

        if not self._load_with_best_layers(hint_layers):
            raise RuntimeError("Nie udało się załadować modelu (ctransformers).")

        # 2) Warmup
        if LLM_WARMUP and self.backend == "ctransformers":
            try:
                _ = self.model(" ", max_new_tokens=4, temperature=0.0, top_p=0.0,
                               threads=self._threads, stop=[])
            except Exception:
                pass

        # 3) Autoboost TYLKO w sesji, max do aktualnego „sufitu”
        session_cap = max(0, self._persist_layers_cap or cfg_cap or 0)
        self._gpu_layers_autoboost_session(upper_cap=session_cap)

        # 4) Prefill batch probe (GPU)
        self._probe_generation_kwargs()

        # 5) Dobór wątków CPU (≥80% best, preferuj mniejsze)
        self._choose_threads_gpu_biased()

        # 6) (opcjonalnie) ponowny prefill po zmianie threads — drobna korekta
        self._probe_generation_kwargs()

        # 7) Zapisz tuning tylko w trybie "on" — PERSIST ONLY DOWN
        self._save_tuning()

    def _autosize_tokens(self, prompt_len: int, user_cap: int) -> int:
        cap = int(user_cap or 512)
        if not LLM_AUTOSIZE_MAX_NEW_TOKENS:
            return cap
        if prompt_len < 4000:
            return cap
        if prompt_len < 8000:
            return min(cap, max(cap, 1200))
        return min(cap, max(cap, 1024))

    def _gen_ctransformers(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
                           stops: Sequence[str], seed: Optional[int]) -> str:
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stop": list(stops),
            "threads": int(self._threads),
        }
        if self._prefill_batch > 0:
            if self._supports_n_batch:
                kwargs["n_batch"] = int(self._prefill_batch)
            elif self._supports_batch_size:
                kwargs["batch_size"] = int(self._prefill_batch)

        try:
            if LLM_TOP_K:
                kwargs["top_k"] = int(LLM_TOP_K)
            if LLM_REPETITION_PENALTY:
                kwargs["repetition_penalty"] = float(LLM_REPETITION_PENALTY)
            if seed is not None:
                kwargs["seed"] = int(seed)
            kwargs["last_n_tokens"] = 8  # lżejsze liczenie powtórzeń
        except Exception:
            pass
        return self.model(prompt, **kwargs)

    def _ends_naturally(self, text: str) -> bool:
        if not text:
            return False
        s = text.rstrip()
        if not s:
            return False
        last = s[-1]
        if last in ".!?…":
            return True
        closing = {'"', "'", ")", "]"}
        if last in closing and len(s) >= 2 and s[-2] in ".!?…":
            return True
        return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMP,
        top_p: float = LLM_TOP_P,
        stop_sequences: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        min_new_tokens: int = 0,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        continue_until_complete: bool = True,
        max_continuations: int = 6,
    ) -> str:
        if self.backend != "ctransformers" or self.model is None:
            raise RuntimeError("LLM niezaładowany (ctransformers).")

        acquired = self._gen_lock.acquire(timeout=180.0)
        if not acquired:
            raise RuntimeError("LLM zajęty zbyt długo (timeout 180s na _gen_lock).")
        try:
            return self._generate_locked(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                seed=seed,
                min_new_tokens=min_new_tokens,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                continue_until_complete=continue_until_complete,
                max_continuations=max_continuations,
            )
        finally:
            self._gen_lock.release()

    def _generate_locked(
        self,
        prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMP,
        top_p: float = LLM_TOP_P,
        stop_sequences: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        min_new_tokens: int = 0,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        continue_until_complete: bool = True,
        max_continuations: int = 6,
    ) -> str:
        temperature = float(max(0.0, min(2.0, temperature if temperature is not None else 0.35)))
        top_p = float(max(0.0, min(1.0, top_p if top_p is not None else 0.95)))

        max_user = int(max_new_tokens or 512)
        max_by_ctx = int(self._ctx_len * 0.6)
        eff_max = int(max(16, min(max_by_ctx, self._autosize_tokens(len(prompt or ""), max_user))))
        stops = tuple(stop_sequences) if stop_sequences is not None else tuple(self._stops)

        text = self._gen_ctransformers(prompt, eff_max, temperature, top_p, stops, seed)
        text = _post_process(text, stops)

        if continue_until_complete:
            total_text = text
            iteration = 0
            while (
                iteration < max_continuations
                and total_text.strip()
                and not self._ends_naturally(total_text)
            ):
                iteration += 1
                cont_prompt = (
                    prompt
                    + total_text
                    + ""
                )
                approx_used = len(cont_prompt) // 3  # ~3 znaki/token (PL)
                room = int(self._ctx_len * 0.9) - approx_used
                if room < 64:
                    break
                cont_max = int(min(eff_max, max(96, room)))

                more = self._gen_ctransformers(
                    cont_prompt, cont_max, temperature, top_p, stops, seed
                )
                more = _post_process(more, stops)

                if not more or not more.strip():
                    break

                joiner = "" if (total_text.endswith(" ") or more.startswith(" ")) else " "
                if total_text and total_text[-1].isalnum() and more[:1].isalnum():
                    joiner = ""
                total_text = (total_text + joiner + more).strip()

            text = total_text

        if LLM_ENABLE_RETRY and not self._ends_naturally(text):
            too_short = len(text) < self._min_chars
            if too_short and temperature <= 0.7:
                text2 = self._gen_ctransformers(
                    prompt,
                    max_new_tokens=min(int(eff_max * 1.2), eff_max + 192),
                    temperature=min(0.7, max(0.35, temperature + 0.10)),
                    top_p=min(0.98, max(0.90, top_p + 0.02)),
                    stops=stops,
                    seed=seed,
                )
                text2 = _post_process(text2, stops)
                if len(text2) > len(text):
                    text = text2

        return text

    def is_busy(self) -> bool:
        acquired = self._gen_lock.acquire(blocking=False)
        if acquired:
            self._gen_lock.release()
            return False
        return True