#!/usr/bin/env python3
"""
Quick setup script for Face Recognition Ensemble project.
This script helps you set up the environment and download necessary dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"⚙️  {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def install_pytorch():
    """Install PyTorch based on system."""
    print("🔥 Installing PyTorch...")
    
    # Check if CUDA is available
    if platform.system() == "Windows":
        # For Windows, try to detect CUDA
        cuda_command = "nvidia-smi"
        try:
            subprocess.run(cuda_command, shell=True, check=True, capture_output=True)
            print("🎮 CUDA detected! Installing PyTorch with CUDA support...")
            torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        except subprocess.CalledProcessError:
            print("💻 No CUDA detected. Installing CPU version...")
            torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else:
        # For Linux/Mac, try CPU first
        print("💻 Installing PyTorch CPU version...")
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(torch_command, "PyTorch installation")

def install_requirements():
    """Install project requirements."""
    requirements_file = Path("requirements.txt")
    
    if requirements_file.exists():
        return run_command("pip install -r requirements.txt", "Installing requirements")
    else:
        # Install essential packages
        packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "opencv-python>=4.5.0",
            "Pillow>=8.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tqdm>=4.62.0",
            "tensorboard>=2.7.0",
            "pyyaml>=6.0",
            "easydict>=1.9",
            "jupyter>=1.0.0",
            "ipython>=7.30.0"
        ]
        
        command = f"pip install {' '.join(packages)}"
        return run_command(command, "Installing essential packages")

def setup_directories():
    """Create necessary directories."""
    print("📁 Setting up directories...")
    
    directories = [
        "data",
        "data/VGGFace2",
        "data/IJB-C",
        "checkpoints",
        "logs",
        "outputs",
        "tensorboard"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directory setup completed!")
    return True

def create_sample_data():
    """Create sample data files for demonstration."""
    print("📝 Creating sample data files...")
    
    # Create sample train list
    train_list = Path("data/VGGFace2/train_list.txt")
    train_list.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_list, 'w') as f:
        for i in range(100):
            f.write(f"person_{i:03d}/img_{i:03d}.jpg person_{i:03d}\n")
    
    # Create sample val list
    val_list = Path("data/VGGFace2/val_list.txt")
    with open(val_list, 'w') as f:
        for i in range(20):
            f.write(f"person_{i:03d}/img_{i:03d}_val.jpg person_{i:03d}\n")
    
    print("✅ Sample data files created!")
    return True

def main():
    """Main setup function."""
    print("🚀 Starting Face Recognition Ensemble Setup...")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install PyTorch
    if not install_pytorch():
        print("⚠️  PyTorch installation failed. You may need to install it manually.")
        print("   Visit: https://pytorch.org/get-started/locally/")
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Requirements installation failed. Please check the error messages above.")
    
    # Setup directories
    if not setup_directories():
        print("⚠️  Directory setup failed.")
    
    # Create sample data
    if not create_sample_data():
        print("⚠️  Sample data creation failed.")
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED!")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("1. 📊 Download VGGFace2 dataset:")
    print("   - Visit: https://github.com/ox-vgg/vgg_face2")
    print("   - Place in: data/VGGFace2/")
    
    print("\n2. 📊 Download IJB-C dataset:")
    print("   - Visit: https://www.nist.gov/programs-projects/face-challenges")
    print("   - Place in: data/IJB-C/")
    
    print("\n3. 🚀 Run the notebook:")
    print("   - Open: notebooks/ensemble_face_recognition.ipynb")
    print("   - Execute cells step by step")
    
    print("\n4. 🏋️ Start training:")
    print("   - Use: python training/train_ensemble.py")
    print("   - Monitor: tensorboard --logdir tensorboard/")
    
    print("\n5. 📈 Evaluate results:")
    print("   - Check IJB-C performance metrics")
    print("   - Compare with baseline models")
    
    print("\n💡 TIPS:")
    print("   - Use GPU for training (CUDA recommended)")
    print("   - Monitor training with TensorBoard")
    print("   - Adjust hyperparameters in config/config.yaml")
    print("   - Use ensemble weights optimization")
    
    print("\n🆘 SUPPORT:")
    print("   - Check README.md for detailed instructions")
    print("   - Review configuration in config/config.yaml")
    print("   - Refer to paper implementations for guidance")
    
    print("\n" + "=" * 60)
    print("Happy face recognition! 🎭")
    print("=" * 60)

if __name__ == "__main__":
    main()
