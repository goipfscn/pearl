# SM80 Noising Addition - Current Status

**SHA**: e35820246cb6f453afe74256824279aff04131b5  
**Build**: PASS 52105432B  
**Sanitizer**: 0 errors (memory safety OK)  
**Numerical**: FAIL (ApEA/BpEB ~170-180 error)

---

## ROOT CAUSE CONFIRMED

`make_tensor_like<T>(tensor)` **allocates NEW storage** (tensor_impl.hpp:418-434), does NOT alias.

```cpp
// WRONG (e3582024):
tCrA_u16 = make_tensor_like<uint16_t>(tCrA);  //  NEW storage
taccCrA = retile_D(tCrA_u16);                 // View of tCrA_u16
// S2R writes taccCrA → updates tCrA_u16
// SM80 addition updates tCrA → DIFFERENT storage (dead write)
// R2S reads taccCrA → reads tCrA_u16 (never sees addition)
```

**Zero-noise test**: EAL=0 → ApEA=A expected, got max_diff=125 → confirms dead write

---

## ADDITIONAL ISSUES

1. **Coordinate mismatch**: Half-K dummy MMA (tCrA) vs full MMA (tCrApEA) have different partition_C coordinates even with same factory name

2. **Layout incompatibility**: LDSM → AutoVectorizing R2S not identity even without noise

3. **A=0 test**: EA isolated shows max_diff=63, EA_ref(0,0)=-42 got 0 → consistent with dead storage

---

## RECOMMENDED APPROACH

**User suggestion**: "Explicit scalar coordinate-addressed SM120 fallback" or "simple separate kernel"

### Option A: Pure SMEM Addition (Simplest)

```cpp
#if SM80
  if (tid < 32) {
    // 1. Store tCrApEA to SMEM using raw 2D addressing
    //    Don't use partition/retile - just write element-by-element
    Element* sEA_ptr = (Element*)sApEA.data();
    for (int i = 0; i < size(tCrApEA_int8); ++i) {
      // Map MMA fragment index to (m,k) coordinate
      // Then write to sEA_ptr[m * stride + k]
    }
    __syncthreads();
    
    // 2. Add in SMEM using flat 2D loops
    Element* sA_ptr = (Element*)sA_pi.data();
    for (int idx = tid; idx < kBlockM * kBlockK; idx += 32) {
      int m = idx / kBlockK;
      int k = idx % kBlockK;
      sEA_ptr[m * stride + k] += sA_ptr[m * stride + k];
    }
    __syncthreads();
    
    // 3. Load result for R2S
    cute::copy(s2r_tiled_copy_A, ...);
  }
#endif
```

### Option B: Separate Kernel

Create `pearl_noising_add_kernel` that just does `C = A + B` on int8 tensors in GMEM.  
Call between MMA and store.

---

## WHAT DOESN'T WORK

❌ Register layout tricks (retile_D, permute, recast aliases)  
❌ Assuming partition_C coordinates match between different MMAs  
❌ Direct element-wise `frag1[i] += frag2[i]` with mismatched layouts

---

## NEXT STEPS

1. Implement Option A with explicit MMA→coordinate mapping
2. OR implement Option B (separate kernel)
3. Test zero-noise (EAL=0), A-zero, B-zero, one-hot
4. Validate exact integer semantics preserved

**Production**: Remains PAUSED until numerical exact
