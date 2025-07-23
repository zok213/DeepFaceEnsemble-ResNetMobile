from setuptools import setup, find_packages

setup(
    name="face_recognition_ensemble",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Face Recognition Ensemble Learning with SE-ResNet-50 and MobileFaceNet",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/face_recognition_ensemble",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "pyyaml>=6.0",
        "easydict>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "cupy-cuda116>=10.0.0",  # Adjust CUDA version as needed
        ],
    },
    entry_points={
        "console_scripts": [
            "train-ensemble=training.train_ensemble:main",
            "evaluate-ensemble=evaluation.evaluate_ensemble:main",
        ],
    },
)
