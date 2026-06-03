# Python 3.12 patterns for `MonarchConfig` + `MonarchBackend`

Scope: single-GPU/HPC tool, Pydantic v2, Python 3.12. Ranked by payoff.
Grounded in `biahub/tile_stitch/config.py` (flat, `extra="forbid"`, `Literal`
fields, no validators yet) and `monarch/backend.py` (manual
`setup → drive_tp → swap → teardown`, no resource guard on exceptions).

---

## ADOPT (ranked)

### 1. `@model_validator(mode="after")` + `typing.Self` for cross-field checks  ⭐ highest
**What:** one instance method that sees fully-coerced fields and enforces rules
spanning more than one field.
**Where:** `MonarchConfig` — the cross-field rules the flat config currently
can't express. e.g. `recon_batch <= prefetch_depth` (the backend comment at
`backend.py:336` literally says prefetch depth should be `>= recon_batch`);
`device == "cpu"` ⇒ `gpus_per_node` must be unset/ignored; `resident_volume`
must fit `gpus_per_node` budget.
```python
from typing import Self
from pydantic import model_validator

@model_validator(mode="after")
def _check(self) -> Self:
    if self.prefetch_depth < self.recon_batch:
        raise ValueError("prefetch_depth must be >= recon_batch for A/B overlap")
    return self
```
**Caveat:** `mode="after"` runs on every construct; keep it pure (no I/O). Must
`return self`.

### 2. Context-manager lifecycle (`__enter__`/`__exit__`) on `MonarchBackend`  ⭐
**What:** make the backend a context manager so `teardown()` runs on any exit,
including exceptions.
**Where:** `backend.py` already has `setup()`/`teardown()` — today a crash in
`drive_tp` leaks the actor mesh (procs only freed on GC). Wrap it:
```python
def __enter__(self) -> Self:
    return self                      # caller still calls setup(first_plan)
def __exit__(self, *exc) -> bool:
    self.teardown()
    return False                     # never swallow
```
Usage: `with MonarchBackend(...) as be: be.setup(p0); ...`. Optionally fold
`setup` into `__enter__` if the first plan path is known at construct time.
**Caveat:** don't return `True` from `__exit__` — that hides driver errors.
`asyncio.run` per call already establishes the boundary; this only guards cleanup.

### 3. `Protocol` for the backend seam + `@typing.override` on impls
**What:** structural interface so `MonarchBackend` (and a future `cpu`/mock
backend, or the dask path) satisfy one contract without an inheritance tree.
**Where:** define `class Backend(Protocol)` with `setup / drive_tp / swap /
teardown`; the dispatcher/CLI types against `Backend`. Add `@override` on the
concrete methods to catch signature drift at type-check time.
```python
from typing import Protocol, runtime_checkable
@runtime_checkable
class Backend(Protocol):
    def setup(self, first_plan_path: str) -> None: ...
    def drive_tp(self, plan_path: str, plan, *, recon_batch: int = 1) -> dict: ...
```
**Caveat:** Protocol over ABC because the dask/legacy path already exists and we
don't want to retrofit a base class onto it. `@override` is a no-op at runtime —
value is in CI/mypy only.

### 4. `enum.StrEnum` for `device` / `compile_mode`
**What:** replace the `Literal[...]` strings (`protocol: Literal["tcp","ucxx"]`,
`pool_mode: Literal["scale","adapt"]`) with `StrEnum` where the value set is
reused outside the model.
**Where:** `device: Device`, `compile_mode: CompileMode`. StrEnum members ARE
strings, so YAML/JSON round-trips with no `use_enum_values`, and `match` (below)
and dispatch code import one symbol instead of duplicating the literals.
```python
from enum import StrEnum
class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
```
**Caveat:** for a value used in exactly one place (e.g. `protocol`), `Literal`
is fine — don't manufacture an enum per field. Adopt only where the set is
referenced from `match`/dispatch too (rule 6).

### 5. `@computed_field` + `functools.cached_property` for derived geometry
**What:** expose derived-once values (and cache them) as part of the model.
**Where:** `MonarchConfig` derived quantities — e.g. effective window
(`window_per_actor * n_gpus`, currently computed inline at `backend.py:242`),
total actors (`nodes * gpus_per_node`), or the cached TF/geometry the task brief
mentions. Moves "compute once from config" out of the driver.
```python
from functools import cached_property
from pydantic import computed_field
@computed_field
@cached_property
def n_actors(self) -> int:
    return self.n_nodes * self.gpus_per_node
```
**Caveat:** `cached_property` needs the model NOT fully frozen, or set it via
`model_config = ConfigDict(frozen=True)` only on the leaf and access caching
carefully — pydantic supports `cached_property` but it writes to `__dict__`, so
test that frozen+cached coexist for your version before committing.

### 6. `match` on the `StrEnum` for backend/transport dispatch
**What:** structural dispatch replacing `if device == ...: elif ...`.
**Where:** the dispatcher selects cpu/gpu-local/gpu-slurm today via the
`*_pool` fields. A `match self.device:` (or the discriminated tag, rule 7) reads
cleaner and is exhaustiveness-checkable.
**Caveat:** only worth it at the one real fork; don't sprinkle `match` over
two-arm conditionals.

### 7. Discriminated union for the pool variants  (medium — only if you collapse the 3 optional pools)
**What:** `CpuSlurmConfig | GpuLocalCudaConfig | GpuSlurmConfig` as a tagged
union on a `cluster_type` discriminator, instead of three `… | None` fields.
**Where:** `TileStitchRun` currently has `cpu_pool/gpu_pool/gpu_slurm_pool` all
optional — nothing stops 0 or 2 being set. A discriminated union makes "exactly
one" structural and gives precise validation errors per variant.
```python
from typing import Annotated, Union
from pydantic import Field
Pool = Annotated[Union[CpuSlurmConfig, GpuLocalCudaConfig, GpuSlurmConfig],
                 Field(discriminator="cluster_type")]
```
**Caveat:** requires adding a `cluster_type: Literal[...]` tag to each variant —
a YAML-shape change. The memo at `config.py` deliberately mirrors the legacy
flat layout; coordinate with the "operators map YAMLs over" goal before doing
this. If you keep three fields, instead add a `model_validator` enforcing
exactly-one (cheaper, no schema break).

---

## SKIP (considered, not worth it here)

- **PEP 695 `type X = ...` aliases / `class C[T]` type params** — config is
  concrete, no generic containers. Pure noise.
- **`@dataclass(slots=True, frozen=True, kw_only=True)` for the config** — we
  ingest YAML from operators ⇒ we need validation/coercion ⇒ Pydantic
  `BaseModel` wins. dataclass is only faster when you skip validation, which we
  can't. (Fine for tiny *internal* structs if any appear, not the config.)
- **`Annotated[int, Field(...)]` everywhere** — current `PositiveInt` /
  `NonNegativeInt` aliases already read well; converting to verbose `Annotated`
  forms buys nothing.
- **Full `model_config = frozen=True` on every config class** — configs are
  built once and not mutated anyway; freeze only if a mutation bug actually
  appears, and watch the `cached_property` interaction (rule 5 caveat).
- **`@contextmanager` generator form for the backend** — the class already holds
  state (`_drained`, `_workers`); class-based `__enter__/__exit__` (rule 2) fits
  the existing object, a generator wrapper would just duplicate it.

---
**Bottom line:** rules 1–3 are the real wins (cross-field safety, leak-free
teardown, a typed seam) and touch existing gaps directly. 4–6 are cheap polish.
7 only if you're willing to change the YAML shape; otherwise a one-of validator.
