#!/usr/bin/env python3
"""
Kaggle-specific setup script for face recognition testing
Optimized for Kaggle environment constraints and pre-installed packages
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def detect_environment():
    """Detect if running in Kaggle environment"""
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    print("🔍 Environment Detection:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Python: {sys.version}")
    print(f"   Environment: {'Kaggle' if is_kaggle else 'Local'}")
    
    if is_kaggle:
        print(f"   Kaggle Kernel Type: {os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Unknown')}")
    
    return is_kaggle

def check_gpu_availability():
    """Check GPU availability and configuration"""
    print("\n🚀 GPU Check:")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ GPU Available: {device_name}")
            print(f"   💾 GPU Memory: {memory_gb:.1f} GB")
            print(f"   🎯 CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("   ❌ GPU not available, will use CPU")
            return False
    except ImportError:
        print("   ❌ PyTorch not available")
        return False

def install_kaggle_requirements():
    """Install packages specifically needed for Kaggle"""
    
    # Essential packages that are often missing in Kaggle
    kaggle_packages = [
        'kagglehub',
        'easydict', 
        'face-recognition',
        'dlib',
        'mtcnn'
    ]
    
    print("\n📦 Installing Kaggle-specific packages:")
    
    for package in kaggle_packages:
        try:
            print(f"   Installing {package}...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package], 
                capture_output=True, 
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"   ✅ {package} installed successfully")
            else:
                print(f"   ⚠️  {package} installation had issues: {result.stderr[:100]}...")
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {package} installation timed out")
        except Exception as e:
            print(f"   ❌ {package} installation failed: {str(e)}")

def verify_installations():
    """Verify that key packages are working"""
    
    print("\n✅ Verification:")
    
    packages_to_check = {
        'numpy': 'import numpy; print(f"NumPy {numpy.__version__}")',
        'torch': 'import torch; print(f"PyTorch {torch.__version__}")',
        'cv2': 'import cv2; print(f"OpenCV {cv2.__version__}")',
        'PIL': 'from PIL import Image; print(f"PIL {Image.__version__}")',
        'kagglehub': 'import kagglehub; print("KaggleHub available")',
        'face_recognition': 'import face_recognition; print("Face Recognition available")',
    }
    
    for package, test_code in packages_to_check.items():
        try:
            exec(test_code)
            print(f"   ✅ {package}: Working")
        except ImportError:
            print(f"   ❌ {package}: Not available")
        except Exception as e:
            print(f"   ⚠️  {package}: Error - {str(e)}")

def setup_kaggle_directories():
    """Create necessary directories for Kaggle environment"""
    
    print("\n📁 Setting up directories:")
    
    directories = [
        '/kaggle/working/outputs',
        '/kaggle/working/models',
        '/kaggle/working/logs',
        '/kaggle/working/cache'
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {str(e)}")

def display_resource_info():
    """Display available system resources"""
    
    print("\n💻 System Resources:")
    
    try:
        # Memory information
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            for line in lines[:3]:
                print(f"   {line.strip()}")
    except:
        print("   Memory info not available")
    
    try:
        # Disk space
        result = subprocess.run(['df', '-h', '/kaggle'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                print(f"   Disk: {lines[1]}")
    except:
        print("   Disk info not available")

def main():
    """Main setup function"""
    
    print("🎭 Kaggle Face Recognition Setup")
    print("=" * 50)
    
    # Detect environment
    is_kaggle = detect_environment()
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Install packages
    install_kaggle_requirements()
    
    # Verify installations
    verify_installations()
    
    # Setup directories (only in Kaggle)
    if is_kaggle:
        setup_kaggle_directories()
        display_resource_info()
    
    # Final summary
    print("\n🎯 Setup Summary:")
    print(f"   Environment: {'Kaggle' if is_kaggle else 'Local'}")
    print(f"   GPU Available: {'Yes' if gpu_available else 'No'}")
    print(f"   Ready for face recognition testing: ✅")
    
    print("\n📝 Next Steps:")
    print("   1. Run the face recognition notebook")
    print("   2. Download VGGFace2 dataset using kagglehub")
    print("   3. Test face detection and recognition")
    
    print("\n🚀 Happy face recognition testing!")

if __name__ == "__main__":
    main()
