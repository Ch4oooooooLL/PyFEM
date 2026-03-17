"""
Smoke test for PINN V2 components - verifies imports without running full training.
"""

import sys
import os

# Add project to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, ROOT_DIR)

def test_imports():
    """Test that all new modules can be imported."""
    try:
        from Deep_learning.models.pinn_loss import AdaptivePINNLoss, UncertaintyAdaptiveLoss
        from Deep_learning.models.pinn_model_v2 import PINNDamagePredictorV2, PINNEnsemble, ResidualBlock
        print("✓ All PINN V2 imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_file_structure():
    """Verify all required files exist."""
    required_files = [
        'Deep_learning/models/pinn_loss.py',
        'Deep_learning/models/pinn_model_v2.py',
        'Deep_learning/train_pinn_v2.py',
        'Deep_learning/configs/pinn_v2.yaml',
        'tests/test_models/test_pinn_loss.py',
        'tests/test_models/test_pinn_v2.py',
        'tests/test_pipeline/test_pinn_v2_training.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(ROOT_DIR, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_config_valid():
    """Test that config file is valid YAML."""
    try:
        import yaml
        config_path = os.path.join(ROOT_DIR, 'Deep_learning/configs/pinn_v2.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        assert 'train' in config, "Missing 'train' section"
        assert 'eval' in config, "Missing 'eval' section"
        print("✓ Config file is valid YAML with required sections")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("PINN V2 Smoke Test")
    print("="*60)
    
    print("\n1. Testing imports...")
    imports_ok = test_imports()
    
    print("\n2. Testing file structure...")
    files_ok = test_file_structure()
    
    print("\n3. Testing config file...")
    config_ok = test_config_valid()
    
    print("\n" + "="*60)
    if imports_ok and files_ok and config_ok:
        print("✓ All smoke tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
