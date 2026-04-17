# Frontend Model Rediscovery Fix

## Basic Information

- Repository: `C:\Users\31248\Desktop\dynamo`
- Branch at fix time: `main`
- Commit: `1c92a306bad144669eebb64671235c3c21a0aa86`
- Modified file: `lib/llm/src/discovery/watcher.rs`

## Problem Background

The issue is the same class of problem discussed before:

1. A new worker comes up during a rolling update or after a backend parameter change.
2. The new worker is not fully registered into `ModelManager` yet.
3. The old worker is then removed.
4. `handle_delete()` sees that other discovery instances for the model still exist, so it returns early.
5. Because it returns early without replaying registration for the still-alive instance, the frontend can temporarily lose the model.

In practice, that shows up as:

- the worker is still present in discovery
- the frontend `/v1/models` cannot see the model
- rebuilding or restarting the frontend causes it to be rediscovered

## Important Difference In This Branch

This branch is not identical to the older branches that were fixed before.

Older branches used a model-wide checksum rule, usually through manager-level validation.

This branch uses per-WorkerSet checksum compatibility:

- code path: `model.is_checksum_compatible(&ws_key, candidate_checksum)`
- meaning: checksum is checked against the existing WorkerSet for the same `ws_key`
- consequence: the reconcile logic must use WorkerSet-level compatibility, not the older model-wide canonical checksum logic

Because of that, this branch must **not** use the older fix form based on `self.manager.is_valid_checksum(...)`.

## What Was Changed

### 1. Import And Helper Type Change

Current file locations:

- import cleanup: `watcher.rs:23`
- new helper type: `watcher.rs:71`

#### Original code

The file imported:

```rust
protocols::{EndpointId, annotated::Annotated},
```

And there was no type carrying full discovery identity for a model instance.

#### New code

The import was changed to:

```rust
protocols::annotated::Annotated,
```

And a new helper struct was added:

```rust
struct DiscoveredModelCard {
    id: ModelCardInstanceId,
    card: ModelDeploymentCard,
}
```

#### Why this was changed

The old helper flow only preserved `EndpointId`, which is not enough to replay `handle_put()`.

To reconstruct a missing registration correctly, the code needs the full `ModelCardInstanceId`, including:

- `namespace`
- `component`
- `endpoint`
- `instance_id`
- `model_suffix`

`EndpointId` became unnecessary after this refactor, so that import was removed.

## 2. `handle_delete()` Was Changed To Reconcile Surviving Instances

Current file locations:

- lookup change: `watcher.rs:317`
- component check change: `watcher.rs:325`
- reconcile call: `watcher.rs:345`

### Original code

`handle_delete()` previously loaded active cards like this:

```rust
let active_instances = self
    .cards_for_model_with_endpoints(&model_name, namespace_filter)
    .await
    .with_context(|| model_name.clone())?;
```

It checked whether the same component still had instances using tuple data:

```rust
let component_has_instances = active_instances.iter().any(|(eid, _)| {
    eid.namespace == *worker_namespace && eid.component == *worker_component
});
```

If `active_instances` was not empty, it only logged and returned:

```rust
if !active_instances.is_empty() {
    tracing::debug!(...);
    return Ok(None);
}
```

### New code

Now it loads instance-aware cards:

```rust
let active_instances = self
    .cards_for_model_instances(&model_name, namespace_filter)
    .await
    .with_context(|| model_name.clone())?;
```

The component check now uses the preserved instance identity:

```rust
let component_has_instances = active_instances.iter().any(|instance| {
    instance.id.namespace == *worker_namespace
        && instance.id.component == *worker_component
});
```

Most importantly, before returning for `!active_instances.is_empty()`, it now runs:

```rust
self.reconcile_active_model_instances(&active_instances)
    .await
    .with_context(|| format!("reconcile active model instances for {model_name}"))?;
```

### Why this was changed

This is the actual fix for the frontend-disappearing-model problem.

Previously:

- old worker removed
- some active discovery instances still existed
- `handle_delete()` returned early
- no one replayed registration for those still-alive instances

Now:

- old worker removed
- if active instances still exist, the watcher proactively reconciles them
- any missing WorkerSet registration can be rebuilt without restarting the frontend

## 3. New `reconcile_active_model_instances()` Helper

Current file location:

- `watcher.rs:837`

### Original code

This function did not exist before.

### New code

A new helper was added:

```rust
async fn reconcile_active_model_instances(
    &self,
    active_instances: &[DiscoveredModelCard],
) -> anyhow::Result<()>
```

Its logic is:

1. Compute `ws_key` from namespace and model type.
2. If the model already has that WorkerSet, skip.
3. If the model exists but the same `ws_key` is checksum-incompatible, skip.
4. Otherwise clone the card and replay `handle_put()` using the preserved `ModelCardInstanceId`.

Key branch-specific checksum guard:

```rust
if let Some(model) = self.manager.get_model(instance.card.name())
    && !model.is_checksum_compatible(&ws_key, instance.card.mdcsum())
{
    ...
    continue;
}
```

### Why this was changed

This helper is what closes the state gap between discovery and `ModelManager`.

It makes deletion idempotent with respect to surviving workers:

- if a WorkerSet is already present, do nothing
- if a WorkerSet is missing but can be safely reconstructed, rebuild it
- if checksum compatibility says the surviving instance should still be rejected, keep rejecting it

That last point matters: the fix restores missing registrations, but it does **not** weaken checksum safety.

## 4. `all_cards()` And Lookup Helpers Were Refactored

Current file locations:

- `all_cards()`: `watcher.rs:873`
- `cards_for_model()`: `watcher.rs:920`
- `cards_for_model_instances()`: `watcher.rs:927`

### Original code

`all_cards()` returned:

```rust
Vec<(EndpointId, ModelDeploymentCard)>
```

The helper below it was:

```rust
async fn cards_for_model_with_endpoints(...)
```

This meant the code only preserved:

- endpoint namespace
- endpoint component
- endpoint name

It lost the full discovery instance identity needed by `handle_put()`.

### New code

`all_cards()` now returns:

```rust
Vec<DiscoveredModelCard>
```

Each item preserves:

```rust
ModelCardInstanceId {
    namespace,
    component,
    endpoint,
    instance_id,
    model_suffix,
}
```

The old helper:

```rust
cards_for_model_with_endpoints(...)
```

was replaced with:

```rust
cards_for_model_instances(...)
```

And `cards_for_model()` now maps:

```rust
.map(|instance| instance.card)
```

instead of:

```rust
.map(|(_, card)| card)
```

### Why this was changed

Without full `ModelCardInstanceId`, the watcher cannot correctly call:

```rust
self.handle_put(&instance.id, &mut card).await?;
```

That is why the old shape was sufficient for filtering, but not sufficient for replaying registration.

## What Was Removed Or Replaced

The following old pieces were removed or replaced:

1. Removed unused import:

```rust
EndpointId
```

2. Replaced helper:

```rust
cards_for_model_with_endpoints(...)
```

with:

```rust
cards_for_model_instances(...)
```

3. Replaced tuple-based component detection:

```rust
|(eid, _)|
```

with instance-aware detection:

```rust
|instance|
```

4. Replaced `all_cards()` return type:

```rust
Vec<(EndpointId, ModelDeploymentCard)>
```

with:

```rust
Vec<DiscoveredModelCard>
```

5. Removed the old behavior where `handle_delete()` would simply return when active instances still existed but no reconciliation had happened.

## What Was Not Changed

The fix was intentionally minimal.

The following behavior was not changed:

- the existing add path in `handle_put()`
- the existing WorkerSet registration path in `do_worker_set_registration()`
- checksum safety rules for same-WorkerSet compatibility
- frontend model update notifications
- prefill/decode WorkerSet key behavior

## Expected Result After This Fix

After this change, when an old worker is removed but a valid new worker is already visible in discovery:

- the watcher can reconstruct the missing WorkerSet registration
- the model remains visible to the frontend
- a frontend restart should no longer be required just to rediscover the model

At the same time, if a surviving instance is still checksum-incompatible for the same WorkerSet, it will continue to be skipped.

## Validation Status

### Completed

- manual source inspection
- manual diff inspection
- commit and push completed successfully

### Not completed in this environment

- `cargo` build
- `rustfmt` check

Reason:

- this machine did not have `cargo` or `rustfmt` available in the shell environment at the time of change

## Short PR Summary

This patch fixes a watcher reconciliation gap in `lib/llm/src/discovery/watcher.rs`.

Before the patch, deleting an old worker could leave the frontend without a visible model even though a new worker was already present in discovery, because `handle_delete()` returned early without replaying registration for surviving instances.

After the patch, the watcher preserves full model instance identity, reconciles still-active instances after deletion, and rebuilds missing WorkerSets when they are checksum-compatible for the current branch's per-WorkerSet validation model.
