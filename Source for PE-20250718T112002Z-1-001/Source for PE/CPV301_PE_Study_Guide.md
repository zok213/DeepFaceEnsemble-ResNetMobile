# CPV301 Practical Exam Study Guide



## Complete Computer Vision & Image Processing Reference

---

## 📚 Table of Contents

1. [Image Fundamentals](#1-image-fundamentals)
2. [Point Operators](#2-point-operators)
3. [Image Smoothing & Filtering](#3-image-smoothing--filtering)
4. [Image Gradients & Edge Detection](#4-image-gradients--edge-detection)
5. [Morphological Transformations](#5-morphological-transformations)
6. [Image Transformations](#6-image-transformations)
7. [Color Spaces & White Balancing](#7-color-spaces--white-balancing)
8. [Feature Detection](#8-feature-detection)
9. [Fourier Transform](#9-fourier-transform)
10. [Practical Exam Tips](#10-practical-exam-tips)
11. [Common Code Patterns](#11-common-code-patterns)

---

## 1. Image Fundamentals

### Basic Image Operations
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('path/to/image.jpg')
gray = cv2.imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)

# Convert BGR to RGB for matplotlib
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display image
plt.imshow(rgb_img)
plt.axis('off')
plt.show()
```

### Image Properties
- **Channels**: Grayscale (1), Color (3: BGR in OpenCV)
- **Data Types**: uint8 (0-255), float32/64 (-∞ to +∞)
- **Shape**: (height, width) for grayscale, (height, width, channels) for color

---

## 2. Point Operators

### 2.1 Alpha Compositing
**Formula**: `C = αF + (1-α)B`

```python
def alpha_compositing(foreground, background, alpha_matte):
    alpha = alpha_matte.astype(float) / 255.0
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    composite = alpha * foreground + (1 - alpha) * background
    return np.clip(composite, 0, 255).astype(np.uint8)
```

### 2.2 Histogram Equalization
**Purpose**: Enhance contrast by redistributing pixel values

```python
def histogram_equalization(gray_img):
    return cv2.equalizeHist(gray_img)
```

**When to use**: Low-contrast images, washed-out images

### 2.3 Thresholding

#### Simple Thresholding
```python
ret, thresh = cv2.threshold(img, thresh_value, max_value, type)
```

**Types**:
- `cv2.THRESH_BINARY`: pixel > thresh → max_value, else → 0
- `cv2.THRESH_BINARY_INV`: pixel > thresh → 0, else → max_value
- `cv2.THRESH_TRUNC`: pixel > thresh → thresh, else → pixel
- `cv2.THRESH_TOZERO`: pixel > thresh → pixel, else → 0
- `cv2.THRESH_TOZERO_INV`: pixel > thresh → 0, else → pixel

#### Adaptive Thresholding
```python
adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize, C)
```

**Parameters**:
- `ADAPTIVE_THRESH_MEAN_C`: Mean of neighborhood - C
- `ADAPTIVE_THRESH_GAUSSIAN_C`: Weighted Gaussian sum - C
- `blockSize`: Size of neighborhood (must be odd)
- `C`: Constant subtracted from mean

#### Otsu's Thresholding
```python
ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**How it works**: Finds optimal threshold by minimizing within-class variance
**Best for**: Bimodal histograms (two distinct peaks)

### 2.4 Contrast Stretching
```python
def contrast_stretching(img):
    return cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
```

---

## 3. Image Smoothing & Filtering

### 3.1 Custom Filtering
```python
# 5x5 averaging kernel
kernel = np.ones((5,5), np.float32) / 25
filtered = cv2.filter2D(img, -1, kernel)
```

### 3.2 Built-in Blurring Techniques

#### Averaging (Box Filter)
```python
blur = cv2.blur(img, (5,5))
# or
blur = cv2.boxFilter(img, -1, (5,5), normalize=True)
```

#### Gaussian Blurring
```python
gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)  # sigmaX=0 (auto-calculate)
```
**Best for**: Removing Gaussian noise

#### Median Blurring
```python
median_blur = cv2.medianBlur(img, 5)  # kernel size must be odd
```
**Best for**: Salt-and-pepper noise

#### Bilateral Filtering
```python
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```
**Best for**: Noise reduction while preserving edges
**Slower**: Much slower than other filters

### Filter Comparison Table

| Filter Type | Best For | Speed | Edge Preservation |
|-------------|----------|-------|-------------------|
| Averaging | General smoothing | Fast | Poor |
| Gaussian | Gaussian noise | Fast | Poor |
| Median | Salt-pepper noise | Medium | Good |
| Bilateral | Edge-preserving smoothing | Slow | Excellent |

---

## 4. Image Gradients & Edge Detection

### 4.1 Gradient Operators

#### Sobel Operator
```python
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # X-direction
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Y-direction
```

#### Scharr Operator (Optimized Sobel)
```python
scharr_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)  # ksize=-1 for Scharr
```

#### Laplacian Operator
```python
laplacian = cv2.Laplacian(img, cv2.CV_64F)
```

### 4.2 Canny Edge Detection
**Most popular edge detection algorithm** - multi-stage process by John F. Canny

#### Theory - 4 Stages:
1. **Noise Reduction**: Apply 5x5 Gaussian filter
2. **Gradient Calculation**: Use Sobel filters to find gradient magnitude and direction  
3. **Non-Maximum Suppression**: Thin edges by keeping only local maxima
4. **Hysteresis Thresholding**: Use two thresholds to classify edges

#### Detailed Process:

**Stage 1: Noise Reduction**
- Edge detection is sensitive to noise
- Apply Gaussian filter to smooth the image
- Reduces false edge detection

**Stage 2: Gradient Calculation**
- Apply Sobel filters in X and Y directions
- Calculate gradient magnitude: `G = √(Gx² + Gy²)`
- Calculate gradient direction: `θ = tan⁻¹(Gy/Gx)`
- Round direction to: 0°, 45°, 90°, 135°

**Stage 3: Non-Maximum Suppression**
- For each pixel, check if it's local maximum in gradient direction
- If pixel is maximum compared to neighbors → keep it
- Else → suppress (set to 0)
- Result: Thin edges

**Stage 4: Hysteresis Thresholding**
- Use two thresholds: `minVal` and `maxVal`
- Pixels > `maxVal` → strong edges (keep)
- Pixels < `minVal` → weak edges (discard)
- Pixels between thresholds → keep only if connected to strong edges

#### Implementation:
```python
# Basic Canny
edges = cv2.Canny(img, minVal, maxVal)

# With preprocessing (recommended)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 100, 200)

# Full parameter version
edges = cv2.Canny(img, minVal, maxVal, apertureSize=3, L2gradient=False)
```

#### Parameter Guidelines:
- **minVal**: Lower threshold (typically 50-100)
- **maxVal**: Upper threshold (typically 2-3x minVal)
- **apertureSize**: Sobel kernel size (3, 5, 7)
- **L2gradient**: True for accurate √(Gx²+Gy²), False for |Gx|+|Gy|
- **Ratio rule**: maxVal should be 2-3 times minVal

#### When to use:
- Need clean, connected edges
- Want to remove noise while preserving important edges
- Standard choice for edge detection in computer vision
- Better than simple gradient operators for noisy images

### 4.3 Important Notes on Data Types

**⚠️ Critical**: Always use `cv2.CV_64F` or `cv2.CV_16S` for gradients!

```python
# Wrong - loses negative values
sobel_wrong = cv2.Sobel(img, cv2.CV_8U, 1, 0)

# Correct approach
sobel_64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel = np.absolute(sobel_64f)
sobel_8u = np.uint8(abs_sobel)
```

### 4.3 Gradient Summary

| Operator | Formula | Purpose |
|----------|---------|---------|
| **Sobel** | Gaussian smoothing + differentiation | Edge detection with noise reduction |
| **Scharr** | Optimized 3x3 Sobel | More accurate than regular 3x3 Sobel |
| **Laplacian** | ∂²I/∂x² + ∂²I/∂y² | Second-order derivative, rapid intensity changes |

---

## 5. Morphological Transformations

### 5.1 Fundamental Operations

#### Erosion
```python
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
```
**Effect**: Shrinks white (foreground) regions
**Use**: Remove small noise, separate connected objects

#### Dilation
```python
dilation = cv2.dilate(img, kernel, iterations=1)
```
**Effect**: Expands white (foreground) regions
**Use**: Join broken parts, restore eroded objects

### 5.2 Compound Operations

#### Opening (Erosion → Dilation)
```python
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
```
**Effect**: Removes small objects (noise) while preserving larger objects

#### Closing (Dilation → Erosion)
```python
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```
**Effect**: Closes small holes inside white regions

#### Morphological Gradient
```python
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```
**Effect**: Difference between dilation and erosion (highlights boundaries)

#### Top Hat
```python
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
```
**Effect**: Difference between original and opening (small bright details)

#### Black Hat
```python
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```
**Effect**: Difference between closing and original (small dark details)

### 5.3 Structuring Elements

```python
# Rectangular
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

# Elliptical
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Cross-shaped
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
```

### Morphological Operations Summary

| Operation | Formula | Effect | Use Case |
|-----------|---------|--------|----------|
| **Erosion** | A ⊖ B | Shrinks foreground | Remove noise |
| **Dilation** | A ⊕ B | Expands foreground | Fill gaps |
| **Opening** | (A ⊖ B) ⊕ B | Remove small objects | Noise removal |
| **Closing** | (A ⊕ B) ⊖ B | Fill small holes | Gap filling |
| **Gradient** | (A ⊕ B) - (A ⊖ B) | Edge detection | Boundary extraction |

---

## 6. Image Transformations

### 6.1 Translation
**Matrix**: 
```
M = [1  0  tx]
    [0  1  ty]
```

```python
tx, ty = 100, 50  # Translation distances
M = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
```

### 6.2 Euclidean (Rigid) Transformation
**Properties**: Preserves angles and distances
```python
center = (cols//2, rows//2)
angle = 45  # degrees
scale = 1.0  # no scaling for pure rotation
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (cols, rows))
```

### 6.3 Similarity Transformation
**Properties**: Preserves angles, allows uniform scaling
```python
angle = 30
scale = 0.75
M = cv2.getRotationMatrix2D(center, angle, scale)
similarity = cv2.warpAffine(img, M, (cols, rows))
```

### 6.4 Affine Transformation
**Properties**: Preserves parallelism, requires 3 point correspondences
```python
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (cols, rows))
```

### 6.5 Projective (Homography)
**Properties**: Most general, preserves straight lines only, requires 4 point correspondences
```python
pts1 = np.float32([[50,50], [200,50], [50,200], [200,200]])
pts2 = np.float32([[10,100], [220,50], [80,250], [200,300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (cols, rows))
```

### Transformation Hierarchy

| Transformation | Points Needed | Preserves | DOF | Use Case |
|----------------|---------------|-----------|-----|----------|
| **Translation** | 1 | Everything | 2 | Simple movement |
| **Euclidean** | 2 | Distances, angles | 3 | Rigid body motion |
| **Similarity** | 2 | Angles, ratios | 4 | Scaled rotation |
| **Affine** | 3 | Parallelism | 6 | Shear, non-uniform scale |
| **Projective** | 4 | Straight lines | 8 | Perspective correction |

---

## 7. Color Spaces & White Balancing

### 7.1 Color Space Conversion
```python
# BGR to RGB (for matplotlib)
rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# BGR to HSV
hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

# BGR to LAB
lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

# BGR to YCrCb
ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)

# BGR to Grayscale
gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
```

### Important Color Spaces:
- **RGB**: Red, Green, Blue - standard for displays
- **HSV**: Hue, Saturation, Value - good for color-based segmentation
- **LAB**: Lightness, A*, B* - perceptually uniform
- **YCrCb**: Luminance and Chrominance - used in JPEG compression

### 7.2 White Balancing Algorithms

#### Gray World Assumption
**Assumption**: Average color of image should be gray
```python
def gray_world_white_balance(img):
    # Calculate channel averages
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    
    # Overall average
    avg = (avg_b + avg_g + avg_r) / 3
    
    # Scaling factors
    scale_b = avg / avg_b
    scale_g = avg / avg_g
    scale_r = avg / avg_r
    
    # Apply scaling
    balanced = np.zeros_like(img, dtype=np.float32)
    balanced[:, :, 0] = img[:, :, 0] * scale_b
    balanced[:, :, 1] = img[:, :, 1] * scale_g
    balanced[:, :, 2] = img[:, :, 2] * scale_r
    
    return np.clip(balanced, 0, 255).astype(np.uint8)
```

#### White Patch Algorithm
**Assumption**: Brightest pixels should be white
```python
def white_patch_white_balance(img, percentile=99):
    # Find percentile values
    max_b = np.percentile(img[:, :, 0], percentile)
    max_g = np.percentile(img[:, :, 1], percentile)
    max_r = np.percentile(img[:, :, 2], percentile)
    
    max_val = 250.0  # Target value
    
    # Scaling factors
    scale_b = max_val / max_b
    scale_g = max_val / max_g
    scale_r = max_val / max_r
    
    # Apply scaling
    balanced = np.zeros_like(img, dtype=np.float32)
    balanced[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    balanced[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    balanced[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)
    
    return balanced.astype(np.uint8)
```

---

## 8. Feature Detection

### 8.1 Understanding Features
**What makes a good feature?**
- **Uniqueness**: Should be distinguishable from surroundings
- **Repeatability**: Should be detectable under different conditions
- **Accuracy**: Should be precisely localized
- **Efficiency**: Should be fast to compute

**Bad features**: Flat regions, repetitive patterns
**Good features**: Corners, edges, blob-like structures

### 8.2 Corner Detection

#### Harris Corner Detection
**Theory**: Based on intensity change in all directions
```python
# Basic Harris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Mark corners
img[corners > 0.01 * corners.max()] = [0, 0, 255]
```

**Parameters**:
- `blockSize`: Window size for corner detection
- `ksize`: Sobel kernel size
- `k`: Harris detector free parameter (0.04-0.06)

#### FAST Corner Detection
**Theory**: Features from Accelerated Segment Test - very fast
```python
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255,0,0))
```

**Advantages**: Extremely fast, good for real-time applications
**Disadvantages**: Not scale or rotation invariant

### 8.3 Feature Descriptors

#### SIFT (Scale-Invariant Feature Transform)
**Theory**: Scale and rotation invariant features
```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
img_with_sift = cv2.drawKeypoints(img, keypoints, None)
```

**Properties**:
- Scale invariant
- Rotation invariant
- Partially illumination invariant
- Slower but very robust

#### SURF (Speeded-Up Robust Features)
**Theory**: Faster alternative to SIFT using box filters
```python
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
keypoints, descriptors = surf.detectAndCompute(gray, None)
```

**Properties**:
- Faster than SIFT
- Uses Hessian matrix for detection
- Uses integral images for speed
- Good balance between speed and robustness

### 8.4 Feature Detection Summary

| Method | Speed | Accuracy | Scale Invariant | Rotation Invariant | Use Case |
|--------|-------|----------|-----------------|-------------------|----------|
| **Harris** | Fast | Good | No | No | Basic corner detection |
| **FAST** | Very Fast | Medium | No | No | Real-time applications |
| **SIFT** | Slow | Excellent | Yes | Yes | High-accuracy matching |
| **SURF** | Medium | Very Good | Yes | Yes | Balanced speed/accuracy |

---

## 9. Fourier Transform

### 9.1 Theory
- **Purpose**: Convert image from spatial domain to frequency domain
- **Low frequencies**: Represent smooth areas, overall structure
- **High frequencies**: Represent edges, details, noise
- **Key insight**: Sharp changes (edges) = High frequency, Smooth regions = Low frequency

### 9.2 Implementation
```python
# Forward FFT
f_transform = np.fft.fft2(gray_img)
f_shift = np.fft.fftshift(f_transform)
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

# Inverse FFT
f_ishift = np.fft.ifftshift(f_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
```

### 9.3 Filtering in Frequency Domain

#### Low-Pass Filter (Remove High Frequencies)
```python
rows, cols = gray_img.shape
crow, ccol = rows//2, cols//2

# Create circular low-pass mask
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), 30, 1, -1)

# Apply mask
fshift = f_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
```

#### High-Pass Filter (Remove Low Frequencies)
```python
# Create high-pass mask (inverse of low-pass)
mask = np.ones((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), 30, 0, -1)

# Apply mask
fshift = f_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
```

### 9.4 Applications
- **Noise removal**: Remove high-frequency noise with low-pass filter
- **Edge enhancement**: Use high-pass filter to emphasize edges
- **Pattern analysis**: Analyze repetitive patterns in frequency domain
- **Image compression**: Remove less important frequency components

---

## 10. Practical Exam Tips

### 10.1 Common Exam Tasks
1. **Image Enhancement**: Histogram equalization, contrast stretching
2. **Noise Reduction**: Gaussian blur, median filter, bilateral filter
3. **Edge Detection**: Sobel, Laplacian, Canny
4. **Thresholding**: Simple, adaptive, Otsu's
5. **Morphological Operations**: Opening, closing, erosion, dilation
6. **Geometric Transformations**: Rotation, scaling, translation, perspective correction
7. **Feature Detection**: Corner detection, keypoint extraction
8. **Color Processing**: Color space conversion, white balancing
9. **Frequency Domain**: FFT, filtering in frequency domain

### 10.2 Problem-Solving Strategy
1. **Understand the problem**: What type of image processing is needed?
2. **Analyze the image**: What are its characteristics (noisy, low contrast, etc.)?
3. **Choose appropriate technique**: Based on image characteristics and desired outcome
4. **Consider preprocessing**: Noise reduction, color conversion if needed
5. **Apply main algorithm**: With appropriate parameters
6. **Post-process if necessary**: Additional filtering, morphological operations
7. **Validate results**: Visual inspection, quantitative measures if available

### 10.3 Parameter Selection Guidelines

#### For Gaussian Blur:
- Small kernel (3x3, 5x5): Light smoothing
- Large kernel (15x15, 21x21): Heavy smoothing

#### For Morphological Operations:
- Small kernel (3x3, 5x5): Fine operations
- Large kernel (7x7, 9x9): Coarse operations

#### For Thresholding:
- Simple: When lighting is uniform
- Adaptive: When lighting varies across image
- Otsu's: When histogram is bimodal

#### For Canny Edge Detection:
- Low thresholds (50, 100): More edges, more noise
- High thresholds (100, 200): Fewer edges, less noise
- Ratio: maxVal should be 2-3x minVal

#### For Feature Detection:
- Harris: Good for basic corner detection
- FAST: Use for real-time applications
- SIFT: Use when you need scale/rotation invariance
- SURF: Good balance of speed and accuracy

---

## 11. Common Code Patterns

### 11.1 Complete Image Processing Pipeline
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def complete_pipeline(image_path):
    # 1. Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Preprocessing
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 3. Main processing (example: edge detection)
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Post-processing
    kernel = np.ones((3,3), np.uint8)
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 5. Display results
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(gray, cmap='gray'), plt.title('Original')
    plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Edges')
    plt.subplot(133), plt.imshow(edges_cleaned, cmap='gray'), plt.title('Cleaned')
    plt.show()
    
    return edges_cleaned
```

### 11.2 Display Helper Function
```python
def show_images(images, titles, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
```

### 11.3 Parameter Comparison Function
```python
def compare_parameters(img, operation, param_list, param_name):
    results = []
    titles = []
    
    for param in param_list:
        if operation == 'gaussian':
            result = cv2.GaussianBlur(img, (param, param), 0)
        elif operation == 'threshold':
            _, result = cv2.threshold(img, param, 255, cv2.THRESH_BINARY)
        # Add more operations as needed
        
        results.append(result)
        titles.append(f'{param_name}={param}')
    
    show_images(results, titles)
```

---

## 12. Debugging Common Issues

### 12.1 Image Loading Problems
```python
# Always check if image loaded successfully
img = cv2.imread('path/to/image.jpg')
if img is None:
    print("Error: Could not load image. Check file path.")
    return
```

### 12.2 Data Type Issues
```python
# Convert data types properly
img_float = img.astype(np.float32) / 255.0  # Normalize to [0,1]
img_uint8 = (img_float * 255).astype(np.uint8)  # Back to [0,255]
```

### 12.3 Dimension Mismatches
```python
# Ensure proper dimensions for operations
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()
```

---

## Quick Reference Cheat Sheet

### Essential OpenCV Functions
- **Loading**: `cv2.imread()`, `cv2.imwrite()`
- **Conversion**: `cv2.cvtColor()`
- **Filtering**: `cv2.blur()`, `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.bilateralFilter()`
- **Gradients**: `cv2.Sobel()`, `cv2.Laplacian()`
- **Thresholding**: `cv2.threshold()`, `cv2.adaptiveThreshold()`
- **Morphology**: `cv2.erode()`, `cv2.dilate()`, `cv2.morphologyEx()`
- **Transforms**: `cv2.warpAffine()`, `cv2.warpPerspective()`

### Key NumPy Functions
- **Array operations**: `np.zeros()`, `np.ones()`, `np.array()`
- **Math**: `np.mean()`, `np.std()`, `np.min()`, `np.max()`
- **Clipping**: `np.clip()`
- **Type conversion**: `.astype()`

### Matplotlib for Display
- **Basic**: `plt.imshow()`, `plt.show()`, `plt.axis('off')`
- **Subplots**: `plt.subplot()`, `plt.figure(figsize=())`
- **Titles**: `plt.title()`, `plt.suptitle()`

---

## Final Exam Preparation Checklist

✅ **Understanding Concepts**
- [ ] Know when to use each filter type
- [ ] Understand the difference between spatial and frequency domain
- [ ] Know the hierarchy of geometric transformations
- [ ] Understand morphological operations and their effects

✅ **Practical Skills**
- [ ] Can load and display images correctly
- [ ] Can apply filters with appropriate parameters
- [ ] Can perform thresholding operations
- [ ] Can implement basic image transformations
- [ ] Can debug common OpenCV issues

✅ **Code Patterns**
- [ ] Know the standard import statements
- [ ] Can write complete processing pipelines
- [ ] Can compare different parameter values
- [ ] Can display results effectively

✅ **Problem Solving**
- [ ] Can analyze image characteristics
- [ ] Can choose appropriate techniques for given problems
- [ ] Can explain the reasoning behind technique selection
- [ ] Can troubleshoot when results don't meet expectations

---

Good luck with your CPV301 practical exam! Remember to:
1. **Read the problem carefully** - understand what's being asked
2. **Start simple** - get basic functionality working first
3. **Test incrementally** - check results after each step
4. **Document your reasoning** - explain why you chose specific techniques
5. **Handle edge cases** - check for common errors and invalid inputs


Course: Computer vision – CPV301
Practical examination
Duration: 90 minutes

Submission: 
Submit your file of code (YourName_YourStudentID.ipynb) and the result (processed images/video).

Requirements:
1. You are required to restore the image with mixing texture (IMG_1.png). It is supposed that you decide to use Fourier transform to remove the texture out of the image. Write a function to do it (built-in fft and ifft are permitted). (3 marks)

2. A customer requires you to stitch 4 images (IMG_2a.jpg, IMG_2b.jpg, IMG_2c.jpg, IMG_2d.jpg) to generate a panorama image. Write a function which uses stitching techniques to combine them into a larger picture. (2 marks)

3. You are required to segment the following image (IMG_3.png). You decide to use K-means to segment the image. However, you do not know how to select the best number K. One method is to compute the average difference between the segmented image and the original one in terms of intensity (pixel by pixel). Please write the function to calculate that average difference, then test with your segmented images using K=2 and K=3.

4. The lighting in the following image (IMG_4.jpg) is insufficient, and the composition could be improved for better balance. You decide to balance it using the “gray world” method. Write a function to implement this method. (2 marks)

i from the abrove and also from the  IMG_2a.jpg, IMG_2b.jpg, IMG_2c.jpg, IMG_2d.jpg in the codespace 
help me to create a notebook to solve it 
must be the notebook format 

check the output.png it got error cause each image it have a different color i think must convert rbg to other to process
You are required to segment the following image (IMG_3.png). You decide to use K-means to segment the image. However, you do not know how to select the best number K. One method is to compute the average difference between the segmented image and the original one in terms of intensity (pixel by pixel). Please write the function to calculate that average difference, then test with your segmented images using K=2 and K=3.
also must be simple word no need to comment to details 
also do not using icon in the notebook make it like a real human work
reasoning and thinking and understanding 
help me do it the best 


