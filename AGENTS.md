# AGENTS.md for `D:\Seafile\PHD\NQS\NetKet\netket`

## Purpose

This file defines the default code-style and implementation constraints for this repository.
When working inside this project, read and follow these rules before making code changes.

## Core Style

1. Keep changes as small and additive as possible.
   - Prefer adding a new file, helper, or thin wrapper over editing multiple existing library files.
   - Do not broaden the scope of a change unless the benchmark or correctness argument requires it.
   - Avoid large refactors that are not directly tied to the task.

2. Stay fully JAX-compatible and efficient.
   - Keep hot paths `jit` / `vmap` friendly.
   - Use fixed-shape arrays and pytrees in performance-sensitive code.
   - Avoid Python callbacks, Python objects, variable-length metadata containers, or host-side branching in critical paths.

3. Match the style of the existing NetKet code.
   - Keep functions short, precise, and minimally layered.
   - Reuse existing helpers, data structures, and calling patterns whenever possible.
   - Avoid verbose abstractions, verbose comments, and redundant wrapper logic.

## Edit Strategy

1. Prefer the smallest viable edit.
2. Prefer reuse over reinvention.
3. Prefer a local helper over a cross-cutting refactor.
4. Prefer a new file over invasive edits when both are reasonable.
5. Modify core operator interfaces only when sampler-side processing is no longer sufficient.

## Performance-Oriented Guidance

1. First remove repeated dense rescans before redesigning interfaces.
2. If a sampler-only optimization can achieve the goal, prefer it before changing operator APIs.
3. If structural speedup requires local metadata to cross the operator-to-sampler boundary, add the narrowest possible capability first.
4. Do not trade away JAX compatibility for convenience.

## Commenting And Structure

1. Comments should be brief and only explain non-obvious intent.
2. Avoid long narrative comments inside hot-path functions.
3. Public helper names should be explicit but not wordy.
4. Keep implementation flow easy to trace from top-level call to kernel.

## Default Decision Rule

When several implementations are possible, prefer the one that is:
1. smaller;
2. more JAX-friendly;
3. more consistent with existing NetKet style;
4. easier to verify for correctness;
5. less invasive to core library files.
