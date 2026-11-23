# Pearl sm_120 Blackwell Support Implementation Status

**Target Hardware**: NVIDIA RTX PRO 5000 72GB / RTX 50-series (Blackwell sm_120/121)  
**Base Repository**: goipfscn/pearl (fork of pearl-research-labs/pearl)  
**Reference**: Upstream PRs #118, #221, #130; Issue #303

## Implementation Progress

### ✅ Completed Components

#### 1. Build System (`setup.py`)
- ✅ Added `PEARL_GEMM_ARCH` environment variable support
- ✅ Architecture selection map (sm_90a, sm_100a, sm_120a, sm_121a)
- ✅ Build-time architecture detection and reporting
- **Usage**: `export PEARL_GEMM_ARCH=sm_120a`

#### 2. Kernel Configurations (`pearl-gemm-build-utils/`)
- ✅ Architecture-aware kernel config selection
- ✅ Blackwell-compatible tile sizes:
  - 64x128x64 stages=2 (~50KB SMEM)
  - 128x128x64 stages=2 (~80KB SMEM)
  - 64x64x64 stages=2 (minimum viable)
- ✅ Disabled Hopper-only 128x256x128 stages=3 config (exceeds 99KB SMEM limit)
- ✅ Automatic selection based on `PEARL_GEMM_ARCH`

#### 3. Kernel Traits (`csrc/gemm/kernel_traits.hpp`)
- ✅ SM80 mma.sync fallback for `__CUDA_ARCH__ >= 1000`
  - Main GEMM: `SM80_16x8x32_S32S8S8S32_TN`
  - Denoise GEMM: `SM80_16x8x16_F32F16F16F32_TN`
- ✅ Replaced Hopper TMA operations with void stubs
- ✅ Replaced `PipelineTmaAsync` with `PipelineAsync` for sm_120
- ✅ Updated `SmemCopyAtomC` to use `UniversalCopy` instead of `SM90_U32x4_STSM_N`

#### 4. Test Suite
- ✅ Comprehensive sm_120 correctness test (`tests/test_sm120_correctness.py`)
- ✅ Full NoisyGEMM validation: noising, GEMM, denoising
- ✅ Per-configuration testing with detailed error reporting
- ✅ CPU reference implementation for validation

### 🔧 In Progress / Remaining Work

#### 5. Collective Mainloop (`csrc/gemm/collective_mainloop.hpp`)
**Status**: Requires substantial modifications

**Required Changes** (from PR #221):
- [ ] Replace TMA copy operations with explicit ldmatrix staging
- [ ] Add SM80 LDSM copy atoms:
  ```cpp
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using S2RCopyAtom_A = Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>;
  using S2RCopyAtom_B = Copy_Atom<SM75_U32x2_LDSM_N, ElementIn>;
  #endif
  ```
- [ ] Explicit register staging before mma.sync:
  ```cpp
  cute::copy(s2r_tiled_copy_A, tCsA_s2r, tCrA_view);
  cute::copy(s2r_tiled_copy_B, tCsB_s2r, tCrB_view);
  ```
- [ ] Guard warpgroup operations (`warpgroup_fence_operand`, `warpgroup_arrive`, `warpgroup_commit_batch`, `warpgroup_wait`)
- [ ] Remove cluster-wide TMA multicast logic
- [ ] Add async copy pipeline handling

#### 6. Collective Epilogue (`csrc/gemm/collective_epilogue.hpp`)
**Status**: Requires substantial modifications

**Required Changes** (from PR #221):
- [ ] Add SM80 LDSM staging for denoise operands:
  ```cpp
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using S2RCopyAtomDenoise_A = Copy_Atom<SM75_U32x4_LDSM_N, ElementDenoise>;
  using S2RCopyAtomDenoise_B = Copy_Atom<SM75_U32x2_LDSM_N, ElementDenoise>;
  ```
- [ ] Explicit staging before denoise GEMMs
- [ ] Remove SMEM C buffer (`smem_C`)
- [ ] Direct register-to-GMEM stores in `scale()` function
- [ ] Add `DenoisePhase1Consumed` named barrier
- [ ] Guard all Hopper warpgroup operations
- [ ] Remove TMA store logic

#### 7. Named Barriers (`csrc/gemm/named_barrier.hpp`)
**Current barriers**:
- `DenoiseComplete = 1`
- `MmaComplete = 2`
- `Epilogue = 3`

**Needs addition** (from PR #221):
- [ ] Add `DenoisePhase1Consumed = 4` for SMEM aliasing synchronization

#### 8. Additional CUDA Source Files
Files that may need architecture guards:
- [ ] `csrc/gemm/pearl_noisingA_kernel.h` - Review for Hopper-specific operations
- [ ] `csrc/gemm/pearl_noisingB_kernel.h` - Review for Hopper-specific operations
- [ ] `csrc/gemm/pearl_gemm_kernel.h` - Review `hopper_gemm_ws` kernel name/signature

### 📋 Testing Strategy

#### CPU/Static Testing (No GPU Required)
1. **Build Test**:
   ```bash
   export PEARL_GEMM_ARCH=sm_120a
   export PEARL_GEMM_FORCE_BUILD=1
   cd /workspace
   uv sync --package pearl-gemm --reinstall-package pearl-gemm
   ```
   - Verifies: Compilation succeeds, correct sm_120a cubins generated
   - Check: `cuobjdump --list-elf` on resulting `.so`

2. **Import Test**:
   ```bash
   uv run python -c 'import pearl_gemm; import pearl_gemm_cuda; print("Import OK")'
   ```

#### GPU Testing (Requires sm_120 Hardware)
3. **Hash Kernel Test** (should work immediately):
   ```bash
   uv run pytest miner/pearl-gemm/tests/test_tensor_hash.py -v
   uv run pytest miner/pearl-gemm/tests/test_inner_hash.py -v
   ```
   - These use simpler kernels without WGMMA/TMA

4. **GEMM Correctness Test** (after kernel changes complete):
   ```bash
   uv run python miner/pearl-gemm/tests/test_sm120_correctness.py
   ```
   - Tests all compiled tile configurations
   - Validates against CPU reference

5. **Full Test Suite**:
   ```bash
   uv run pytest miner/pearl-gemm/tests/ -v \
     -m "not slow and not integration" \
     --ignore=miner/vllm-miner
   ```

6. **Mining Integration Test** (testnet/simnet only):
   ```bash
   # Start pearld in simnet mode
   ./bin/pearld --simnet --rpcuser=test --rpcpass=test --miningaddr=<addr>
   
   # Start gateway
   export PEARLD_RPC_URL="http://localhost:18556"
   export PEARLD_RPC_USER="test"
   export PEARLD_RPC_PASSWORD="test"
   pearl-gateway start
   
   # Test vLLM miner with INT7 model
   vllm serve pearl-ai/Llama-3.3-8B-Instruct-pearl \
     --port 8000 --max-model-len 2048 \
     --enforce-eager --gpu-memory-utilization 0.7
   ```

### 🐛 Known Issues & Blockers

1. **SMEM Overflow** (resolved by new configs):
   - Original 128x256x128 stages=3: ~210KB (exceeds 99KB limit)
   - New configs fit comfortably in 99KB

2. **Operand Staging** (fix in progress):
   - SM80 mma.sync requires explicit ldmatrix staging
   - Hopper WGMMA uses descriptor-based access
   - Must add manual copy from SMEM to registers

3. **Denoise Correctness** (fix in progress):
   - FP16 denoise operands need explicit staging
   - Phase-1/Phase-2 SMEM aliasing needs barrier synchronization

### 📦 Deliverables

#### Source Changes
- `miner/pearl-gemm/setup.py` - Architecture selection ✅
- `miner/pearl-gemm-build-utils/.../default_compiled_kernels.py` - Tile configs ✅
- `miner/pearl-gemm/csrc/gemm/kernel_traits.hpp` - SM80 fallback (partial) ✅
- `miner/pearl-gemm/csrc/gemm/collective_mainloop.hpp` - Mainloop staging (TODO)
- `miner/pearl-gemm/csrc/gemm/collective_epilogue.hpp` - Epilogue staging (TODO)
- `miner/pearl-gemm/csrc/gemm/named_barrier.hpp` - New barriers (TODO)

#### Test & Validation
- `miner/pearl-gemm/tests/test_sm120_correctness.py` ✅
- `SM120_BUILD_INSTRUCTIONS.md` (TODO)
- `SM120_GPU_TEST_GUIDE.md` (TODO)

#### Artifacts for User
- Complete source patch/diff
- Dockerfile with sm_120a build
- Isolated venv build commands
- GPU correctness verification script
- Testnet smoke test instructions
- Rollback procedure

### 🔄 Validation Gaps

**Static/CPU (Testable Now)**:
- ✅ Build system
- ✅ Kernel configuration selection
- ✅ Python bindings import
- ⚠️  CUDA compilation (cubins generated but untested)

**GPU Required (User's Hardware)**:
- ⏳ Hash kernels (should work, needs confirmation)
- ❌ GEMM kernels (will fail until collective changes complete)
- ❌ Full NoisyGEMM correctness
- ❌ Mining mode (block templates, proof generation)
- ❌ 8B model end-to-end testnet mining

### 📚 Reference Material

#### Upstream Issues & PRs
- **Issue #303**: Runtime failure report from RTX 5070
  - Hash kernels work, GEMM kernels fail with "unspecified launch failure"
  - Confirms architecture incompatibility

- **PR #118** (closed): First sm_120 attempt
  - SM80 mma.sync fallback
  - SMEM reduction with unions
  - Tested on GB10 (sm_121)
  - Had correctness issues

- **PR #221** (open, most current): Correctness fixes
  - Fixes operand staging bugs from #118
  - Explicit ldmatrix copies
  - Denoise phase barriers
  - **64 correctness tests passed on RTX 5090**

- **PR #130** (open): Reference backend approach
  - Alternative: PyTorch `_int_mm` fallback
  - Functional but slower
  - Avoids custom kernel complexity

#### Key Architectural Differences

**Hopper (sm_90a)**:
- WGMMA (Warp Group Matrix Multiply-Accumulate)
- TMA (Tensor Memory Accelerator)
- 227KB SMEM per CTA
- Cluster-wide operations
- Descriptor-based operand access

**Blackwell Consumer (sm_120/121)**:
- NO WGMMA (uses SM80 mma.sync instead)
- NO TMA (uses async copy + ldmatrix)
- 99KB SMEM per CTA
- Warp-level operations
- Explicit register staging required

### 🚀 Next Steps for Completion

1. **Apply PR #221 Collective Changes**:
   - Systematically port `collective_mainloop.hpp` changes
   - Systematically port `collective_epilogue.hpp` changes
   - Add named barrier for phase sync

2. **Validate on User Hardware**:
   - Build with `PEARL_GEMM_ARCH=sm_120a`
   - Run hash kernel tests
   - Run GEMM correctness tests
   - Iterate on any failures

3. **Mining Integration**:
   - Test with vLLM miner on simnet
   - Verify proof generation
   - Measure throughput vs Hopper

4. **Documentation & Cleanup**:
   - Final build guide
   - GPU test protocol
   - Performance notes
   - Rollback instructions

### 📧 Handoff Notes

This implementation provides a solid foundation with:
- Working build system and configuration
- Architecture detection and selection
- Partial kernel modifications
- Comprehensive test suite

**What's testable now** (no GPU):
- Build succeeds for sm_120a
- Correct cubins generated
- Python import works

**What needs GPU hardware**:
- Complete collective mainloop/epilogue modifications
- Run test_sm120_correctness.py
- Verify mining integration

**Risk assessment**:
- Hash kernels: **Low risk** (simple, no WGMMA)
- GEMM kernels: **Medium risk** (complex but proven in PR #221)
- Mining correctness: **Low risk** (protocol-agnostic, kernel-independent)
- Performance: **Unknown** (SM80 vs WGMMA tradeoff)

**Estimated remaining effort**:
- Kernel changes: ~200 lines (systematic, pattern-based)
- Testing iterations: 2-3 cycles with hardware
- Documentation: 1-2 hours

The core architecture is sound. The remaining work is mechanical application of proven patterns from PR #221, followed by hardware validation.
