#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
Run this to diagnose import issues
"""

import sys
import os

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            __import__(package_name)
            print(f"✅ {package_name} imported successfully")
        else:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ {module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} import error: {e}")
        return False

def main():
    print("🔧 Testing PPE Detection Module Imports")
    print("=" * 50)
    
    # Test core dependencies
    core_modules = [
        ('flask', 'flask'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'pillow'),
        ('smtplib', None),  # Built-in
        ('threading', None),  # Built-in
        ('datetime', None),  # Built-in
        ('time', None),  # Built-in
        ('os', None),  # Built-in
        ('imghdr', None),  # Built-in
    ]
    
    # Test optional dependencies
    optional_modules = [
        ('pywhatkit', 'pywhatkit'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
    ]
    
    print("\n📦 Testing Core Dependencies:")
    core_success = 0
    for module, package in core_modules:
        if test_import(module, package):
            core_success += 1
    
    print("\n📦 Testing Optional Dependencies:")
    optional_success = 0
    for module, package in optional_modules:
        if test_import(module, package):
            optional_success += 1
    
    # Test local module
    print("\n📦 Testing Local Module:")
    try:
        from model_inference import PPEModel
        print("✅ model_inference.PPEModel imported successfully")
        local_success = True
    except ImportError as e:
        print(f"❌ model_inference import failed: {e}")
        local_success = False
    except Exception as e:
        print(f"⚠️ model_inference import error: {e}")
        local_success = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Import Test Summary:")
    print(f"Core dependencies: {core_success}/{len(core_modules)}")
    print(f"Optional dependencies: {optional_success}/{len(optional_modules)}")
    print(f"Local module: {'✅' if local_success else '❌'}")
    
    if core_success == len(core_modules) and local_success:
        print("\n🎉 All critical imports successful! The app should work.")
    elif core_success >= len(core_modules) - 2 and local_success:
        print("\n⚠️ Most imports successful. Some optional features may not work.")
    else:
        print("\n❌ Critical imports failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")
    
    # Check model file
    print("\n📁 Checking Model Files:")
    model_path = 'runs/train_ppe/ppe_detector/weights/best1.pt'
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model weights are in the correct location")

if __name__ == "__main__":
    main()

