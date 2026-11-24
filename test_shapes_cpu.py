#!/usr/bin/env python3
"""CPU-only shape validation - no GPU needed."""

def test_shapes():
    """Verify matrix operation shapes match expected dimensions."""
    # Simulate _int_mm as regular matmul
    def _int_mm(A, B):
        """torch._int_mm(A, B) = A @ B"""
        import numpy as np
        return np.matmul(A, B)
    
    import numpy as np
    
    # Non-square to expose bugs
    m, n, k, R = 128, 96, 192, 64
    print(f"Testing shapes: m={m}, n={n}, k={k}, R={R} (k≠R≠m≠n)\n")
    
    # Create dummy tensors
    A = np.random.randint(-64, 64, (m, k), dtype=np.int8)
    B = np.random.randint(-64, 64, (n, k), dtype=np.int8)
    EAL = np.random.randint(-64, 64, (m, R), dtype=np.int8)
    EBR = np.random.randint(-64, 64, (n, R), dtype=np.int8)
    EAR_R_major = np.random.randint(-1, 2, (k, R), dtype=np.int8)  # sparse
    EBL_R_major = np.random.randint(-1, 2, (k, R), dtype=np.int8)
    
    print("Tensor shapes:")
    print(f"  A: {A.shape}")
    print(f"  B: {B.shape}")
    print(f"  EAL: {EAL.shape}")
    print(f"  EBR: {EBR.shape}")
    print(f"  EAR_R_major: {EAR_R_major.shape}")
    print(f"  EBL_R_major: {EBL_R_major.shape}")
    print()
    
    # Test A operations
    print("Noise A operations:")
    try:
        EA = _int_mm(EAL, EAR_R_major.T)  # (m,R) @ (R,k) = (m,k)
        print(f"  ✓ EA = EAL @ EAR_R_major.T: {EA.shape} (expect ({m},{k}))")
        assert EA.shape == (m, k), f"Expected ({m},{k}), got {EA.shape}"
        
        ApEA = A + EA
        print(f"  ✓ ApEA = A + EA: {ApEA.shape}")
        
        AxEBL = _int_mm(A, EBL_R_major)  # (m,k) @ (k,R) = (m,R)
        print(f"  ✓ AxEBL = A @ EBL_R_major: {AxEBL.shape} (expect ({m},{R}))")
        assert AxEBL.shape == (m, R)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    
    # Test B operations  
    print("Noise B operations:")
    try:
        EB = _int_mm(EBR, EBL_R_major.T)  # (n,R) @ (R,k) = (n,k)
        print(f"  ✓ EB = EBR @ EBL_R_major.T: {EB.shape} (expect ({n},{k}))")
        assert EB.shape == (n, k)
        
        BpEB = B + EB
        print(f"  ✓ BpEB = B + EB: {BpEB.shape}")
        
        EARxBpEB = _int_mm(BpEB, EAR_R_major)  # (n,k) @ (k,R) = (n,R)
        print(f"  ✓ EARxBpEB = BpEB @ EAR_R_major: {EARxBpEB.shape} (expect ({n},{R}))")
        assert EARxBpEB.shape == (n, R)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False
    
    print()
    print("✓ ALL SHAPES VALID")
    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if test_shapes() else 1)
