# Phase 4: Parallel Data Generation - Plan Verification

**Verified:** 2026-03-17
**Verdict:** PASS_WITH_NOTES

---

## 1. Goal-Backward Analysis

### Success Criteria Mapping

| criterion | Plan Coverage | Task(s) | Assessment |
|-----------|---------------|---------|-------------|
| 4-core speedup 3-3.5x | multiprocessing.Pool implementation with imap_unordered | Task 4,5 | Adequate |
| 20,000 samples <10 min | In-memory collection, parallel workers | Task 5 | Adequate |
| Original interface preserved | Default parameters, backward-compatible signature | Task 2,6 | Adequate |
| --jobs/-j parameter | CLI argparse addition | Task 6 | Adequate |
| --seq flag | CLI argparse addition | Task 6 | Adequate |

### Goal Achievement Path

```
Goal: 3-3.5x speedup
  └── Task 5: Pool.imap_unordered parallel execution
      └── Task 4: _generate_sample worker function (Windows picklable)
          └── Task 3: Sequential logic extraction (refactor)
              └── Task 2: Function signature update
                  └── Task 7: Import additions (parallel)

Goal: Backward compatibility
  └── Task 2: n_jobs=-1, sequential=False defaults
      └── Task 6: CLI args with sensible defaults
          └── Verification: Unit + integration tests (Task 8)
```

**Assessment:** Goal-path is complete. All success criteria map to specific implementation tasks.

---

## 2. Completeness Check

### Required Tasks Analysis

| Task | Included? | Notes |
|------|-----------|-------|
| Worker function (picklable) | Yes | Task 4 - top-level function, tuple args |
| Parallel pool execution | Yes | Task 5 - Pool with imap_unordered |
| Progress display | Yes | Task 5 - tqdm wrapper |
| Result sorting/assembly | Yes | Task 5 - sorted by sample_idx |
| CLI arguments | Yes | Task 6 - --jobs/-j, --seq |
| Import additions | Yes | Task 7 - multiprocessing, tqdm |
| Dependency addition | Yes | Task 1 - tqdm to requirements.txt |
| Tests | Yes | Task 8 - determinism, small-scale, CLI |

### Missing Items

| Item | Severity | Recommendation |
|------|----------|----------------|
| Metadata saving in parallel mode | Medium | `_save_metadata()` called but not defined in plan |
| `_build_random_load_specs` pass-through | Low | Verified - worker receives all needed config |
| Error cleanup on worker failure | Low | PLAN states "fail-fast" - acceptable |

---

## 3. Dependency Validation

### Task Dependency Graph

```
Task 1 (tqdm dep) ─────────────────────────────────────┐
Task 2 (signature) ───> Task 3 (extract seq) ───> Task 4 (worker) ───> Task 5 (parallel) ───> Task 6 (CLI)
Task 7 (imports) ──────────────────────────────────────┘
                                                                       │
Task 8 (tests) ────────────────────────────────────────────────────────┘
```

### Validation

| Dependency | Correct? | Notes |
|------------|----------|-------|
| Task 3 depends on Task 2 | Yes | Need signature before extracting |
| Task 4 depends on Task 3 | Yes | Worker calls extracted logic |
| Task 5 depends on Task 4 | Yes | Pool uses worker function |
| Task 6 depends on Task 5 | Yes | CLI needs parallel path |
| Task 8 depends on Tasks 1-6 | Yes | Tests require complete implementation |
| Tasks 1, 2, 7 parallel | Yes | Independent modifications |

**Assessment:** Dependency order is correct. No circular dependencies.

---

## 4. Risk Assessment

### High-Impact Risks

| Risk | Probability | Impact | Mitigation in Plan |
|------|-------------|--------|-------------------|
| Windows spawn incompatibility | Low | High | Yes - top-level function, tuple args |
| Memory exhaustion (20K samples) | Low | Medium | Partial - in-memory approach noted |
| Determinism violation | Medium | High | Yes - seed derivation formula documented |
| Break existing callers | Low | High | Yes - default parameters preserved |

### Medium-Impact Risks

| Risk | Probability | Impact | Notes |
|------|-------------|--------|-------|
| tqdm not available | Low | Low | Task 1 adds to requirements |
| Worker exception not propagated | Low | Medium | PLAN: "fail-fast" uses default pool behavior |
| Performance underachieving | Medium | Medium | Need benchmark validation |

### Risk Mitigation Recommendations

1. **Memory Consideration:** For very large datasets (>50K samples), consider chunked processing or temp files. Current 20K target should fit in 16GB RAM.

2. **Benchmark Test:** Task8 should include explicit timing test: 1000 samples with jobs=1 vs jobs=4 to verify speedup ratio.

---

## 5. Gap Analysis

### Implementation Gaps

| Gap | Location | Resolution |
|-----|----------|------------|
| `_save_metadata()` function definition | Task 5 calls undefined function | Extract from current data_gen.py:425-450 or inline the code |
| Progress output format change | Current: print every 100 samples → Plan: tqdm | Acceptable change, document in release notes |
| Metadata collection in parallel | Need to aggregate metadata from workers | Current plan handles this correctly - results collected then metadata saved |

### Documentation Gaps

| Gap | Recommendation |
|-----|----------------|
| No mention of `--seq` behavior when n_jobs=1 | Clarify: sequential=True OR n_jobs==1 both trigger sequential mode |
| Test configuration file location | Specify path for test_config.yaml |

### Edge Cases

| Case | Coveredin Plan? | Notes |
|------|----------------|-------|
| n_jobs > cpu_count | Implicitly handled by Pool | Pool handles gracefully |
| n_jobs <= 0 | Yes - defaults to cpu_count-1 | Task 2 covers this |
| num_samples not divisible by n_jobs | Yes - imap_unordered handles | Results sorted after collection |
| Empty dataset (num_samples=0) | Not specified | Should add validation |

---

## 6. Discrepancy Notes

### CONTEXT.md vs PLAN.md

| Decision | CONTEXT.md | PLAN.md | Assessment |
|----------|------------|---------|------------|
| Result merging | "临时文件 + 最终合并" | In-memory collection | PLAN chose simpler approach - acceptable |
| Worker output | Worker writes temp file | Worker returns tuple | PLAN chose cleaner design - correct |

The PLAN.md approach is valid and simpler than the CONTEXT.md decision. Recommend updating CONTEXT.md to reflect final choice.

---

## 7. Test Coverage Assessment

### Covered by PLAN

| Test Type | Coverage | Task |
|-----------|----------|------|
| Determinism | Parallel vs sequential output compare | Task 8 |
| Small-scale execution | samples=10, jobs=2 | Task 8 |
| CLI parsing | --jobs/-j, --seq arguments | Task 8 |
| Speedup benchmark | 1000 samples speedup test | **MISSING** |

### Recommended Additions

```python
# Should add to Task 8:
def test_speedup_ratio():
    """Verify at least 2x speedup on 4-core machine."""
    import time
    
    # Sequential timing
    start= time.time()
    generate_dataset(config_path="test_config.yaml", n_jobs=1, sequential=True)
    seq_time = time.time() - start
    
    # Parallel timing (n_jobs=4)
    start = time.time()
    generate_dataset(config_path="test_config.yaml", n_jobs=4, sequential=False)
    par_time = time.time() - start
    
    speedup = seq_time / par_time
    assert speedup >= 2.0, f"Speedup {speedup:.2f}x < 2.0x minimum"
```

---

## 8. Verification Steps Assessment

### PLAN Verification Checklist

| Category | Adequate? | Gaps |
|----------|-----------|------|
| Unit Tests | Yes | - |
| Integration Tests | Yes | - |
| Performance Tests | Partial | Missing explicit speedup ratio test |
| Backward Compatibility | Yes | - |

---

## 9. Recommendations

### Before Implementation

1. **Add `_save_metadata()` definition** - Either extract to helper function or inline the code in Task 5
2. **Add speedup ratio test** - Explicit benchmark in Task 8
3. **Clarify sequential mode logic** - Document: `sequential=True or n_jobs==1`
4. **Add num_samples=0 validation** - Edge case handling

### Implementation Notes

1. Task 7 imports can be at module level or function-level (PLAN shows both -recommend module-level for clarity)
2. Progress output will change from "进度: n/m" to tqdm - acceptable UX improvement
3. Metadata file saving should happen after all results collected (correct in PLAN)

---

## 10. Verdict

**PASS_WITH_NOTES**

### Summary

The plan is fundamentally sound and will achieve the phase goal. The parallel execution strategy using `multiprocessing.Pool` with `imap_unordered` is appropriate for the embarrassingly parallel FEM sample generation workload. Windows compatibility is properly addressed through top-level worker functions and tuple arguments.

### Critical Items Addressed

- Goal achievement: All success criteria map to implementation tasks
- Dependency order: Correct and properly sequenced
- Windows compatibility: Handled via picklable worker function
- Backward compatibility: Preserved through default parameters

### Minor Items Requiring Attention

1. Define `_save_metadata()` helper function or inline the code
2. Add explicit speedup ratio test to Task 8
3. Document sequential mode trigger conditions clearly
4. Consider updating CONTEXT.md to reflect in-memory approach decision

### Confidence Level

**High** - The plan is ready for implementation with minor clarifications noted above.

---

*Checked by: gsd-plan-checker agent*
*Date: 2026-03-17*