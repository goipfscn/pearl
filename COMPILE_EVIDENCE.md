# SHA 0d4ced511d2731d5eea7dffa14c4ebb823b154b6

## PREPROCESSOR CORRECTNESS

Fixed 37a95fbb compile failures:
- Error: `expected ';' before store_AxEBL template` cascade A778/837/840/847
- Root: Stray `}` when `__CUDA_ARCH__` undefined (host compilation path)
- Root: sync(32) called by all 128 threads outside guard

## PATTERN (noisingA:729-759, noisingB:745-779)

###BEFORE (37a95fbb - BROKEN):
```cpp
#if __CUDA_ARCH__ >= 1000
  if (tid < 32) {    // Open on device
#endif
    copy(...);
#if __CUDA_ARCH__ < 1000
  sync(128);         // SM90 path
#else
  }                  // ← STRAY } on host (__CUDA_ARCH__ undefined)
  sync(32);          // ← OUTSIDE guard, all 128 call it
#endif
  fence();
```

### AFTER (0d4ced51 - CORRECT):
```cpp
#if __CUDA_ARCH__ >= 1000
  if (tid < 32) {       // Open on device
#endif
    copy(...);
#if __CUDA_ARCH__ >= 1000
    sync(32);           // SM80: INSIDE guard, 32 callers
#else
    sync(128);          // SM90: 128 callers
#endif
    fence();            // INSIDE guard on SM80
#if __CUDA_ARCH__ >= 1000
  }                     // Close on device
#endif
  consumer_release();   // Unguarded (all threads, pipeline FSM)
```

## VERIFICATION

### Preprocessor Syntax (host + device paths)
```bash
# Test host (__CUDA_ARCH__ undefined)
g++ -std=c++17 -c test_preprocessor.cpp -o test_host.o
# ✓ Compiles cleanly

# Test device (__CUDA_ARCH__ = 1200)
g++ -std=c++17 -D__CUDA_ARCH__=1200 -c test_preprocessor.cpp -o test_device.o
# ✓ Compiles cleanly
```

### File Hashes
```
150ded3cb942396b825f96ebba90a52d9ffccbd4b8b50ae7285ed7bdc33a4261  pearl_noisingA_kernel.h
bb0111c38ffbf630015267c380bbb5154e16891a06e4c13062e62517d584f7bb  pearl_noisingB_kernel.h
```

## REMAINING TESTING (requires hardware)

1. **Full CUDA compile** (needs nvcc + CUDA toolkit)
   ```bash
   cd pearl/miner/pearl-gemm
   export PEARL_GEMM_ARCH=sm_120a PEARL_GEMM_FORCE_BUILD=1 MAX_JOBS=4
   rm -rf build *.so
   python3 setup.py build_ext --inplace
   ```

2. **Numerical validation** (needs sm_120 GPU)
   ```bash
   python3 diagnose_noising_sm120.py
   # Expected: ApEA=0 AxEBL=0 BpEB=0 EARxBpEB=0
   ```

3. **Memory sanitizer** (needs compute-sanitizer)
   ```bash
   compute-sanitizer --tool memcheck --error-exitcode 2 \
     python3 diagnose_noising_sm120.py
   # Expected: clean, no OOB errors
   ```

## STATUS

- [x] Preprocessor balanced (host + device paths verified)
- [x] Barrier participant-safe (sync count matches callers)
- [x] S2R guards complete (MMA inputs + A/B loads)
- [x] R2S guards complete (outputs)
- [x] Addition loops guarded
- [x] Pipeline ops unguarded (FSM correctness)
- [ ] CUDA device compile (needs GPU hardware)
- [ ] Numerical correctness (needs sm_120)
- [ ] Sanitizer clean (needs compute-sanitizer)

## UNCHANGED FROM 1f143f9d

- TiledMma types (SM80 atoms for all 6 MMA sites)
- MMA operand S2R guards (1f143f9d)
- Output R2S guards (cbe50ce1/1f143f9d)
- Diagnostic shapes (4853acc2)
- Configuration checks

## PRODUCTION

**PAUSED** - DO NOT RESTORE

No mainnet operations, no mining, no proof submission.
