# CPV301 PE Quick Reference Card

## 🚀 Essential Imports
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

## 📖 Image Loading & Display
```python
# Load
img = cv2.imread('path.jpg')  # Color
gray = cv2.imread('path.jpg', cv2.IMREAD_GRAYSCALE)  # Grayscale

# Convert BGR→RGB for matplotlib
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display
plt.imshow(rgb)  # or plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()
```

## 🎯 Point Operations Quick Reference

### Thresholding
```python
# Simple
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Adaptive
adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, 11, 2)

# Otsu's
ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Enhancement
```python
# Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Contrast Stretching
stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
```

## 🌊 Smoothing Filters

| Filter | Code | Best For |
|--------|------|----------|
| **Averaging** | `cv2.blur(img, (5,5))` | General smoothing |
| **Gaussian** | `cv2.GaussianBlur(img, (5,5), 0)` | Gaussian noise |
| **Median** | `cv2.medianBlur(img, 5)` | Salt-pepper noise |
| **Bilateral** | `cv2.bilateralFilter(img, 9, 75, 75)` | Edge-preserving |

## ⚡ Gradient Operators

```python
# Always use CV_64F for gradients!
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Convert back to uint8
abs_sobel = np.absolute(sobel_x)
sobel_8u = np.uint8(abs_sobel)
```

## 🔍 Edge Detection

```python
# Canny Edge Detection (most popular)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 100, 200)  # minVal, maxVal

# Parameter rule: maxVal = 2-3 × minVal
```

## 🔧 Morphological Operations

```python
kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)          # Shrink
dilation = cv2.dilate(img, kernel, iterations=1)        # Expand
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # Remove noise
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)# Fill holes
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) # Edges
```

## 🔄 Transformations

### Translation
```python
M = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M, (width, height))
```

### Rotation
```python
center = (width//2, height//2)
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (width, height))
```

### Affine (3 points)
```python
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (width, height))
```

### Perspective (4 points)
```python
pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (300, 300))
```

## 🎨 Color Processing

```python
# Color space conversion
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# White balancing (Gray World)
avg_b, avg_g, avg_r = np.mean(img, axis=(0,1))
avg = (avg_b + avg_g + avg_r) / 3
scale_b, scale_g, scale_r = avg/avg_b, avg/avg_g, avg/avg_r
balanced = img * [scale_b, scale_g, scale_r]
balanced = np.clip(balanced, 0, 255).astype(np.uint8)
```

## 🔍 Feature Detection

```python
# Harris Corners
gray = np.float32(gray)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# FAST Features
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)

# SIFT Features
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
```

## 📊 Frequency Domain

```python
# FFT
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude = 20 * np.log(np.abs(fshift))

# Low-pass filter
mask = np.zeros_like(gray)
cv2.circle(mask, (center_x, center_y), radius, 1, -1)
filtered = fshift * mask
result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
```

## 🎯 Common Exam Patterns

### Problem Analysis Checklist
1. **What type of noise?** → Choose appropriate filter
2. **What features to extract?** → Choose detection method  
3. **Binary or grayscale output?** → Thresholding needed?
4. **Geometric correction needed?** → Transformation type?

### Technique Selection Guide

| Problem | Technique | Code Pattern |
|---------|-----------|-------------|
| **Noisy image** | Gaussian/Median blur | `cv2.GaussianBlur()` / `cv2.medianBlur()` |
| **Low contrast** | Histogram equalization | `cv2.equalizeHist()` |
| **Uneven lighting** | Adaptive threshold | `cv2.adaptiveThreshold()` |
| **Extract objects** | Thresholding + morphology | `cv2.threshold()` + `cv2.morphologyEx()` |
| **Find edges** | Canny edge detection | `cv2.Canny()` |
| **Find corners** | Harris or FAST | `cv2.cornerHarris()` / `cv2.FastFeatureDetector_create()` |
| **Match features** | SIFT or SURF | `cv2.SIFT_create()` / `cv2.xfeatures2d.SURF_create()` |
| **Correct perspective** | Perspective transform | `cv2.getPerspectiveTransform()` |
| **Remove noise (frequency)** | FFT + low-pass filter | `np.fft.fft2()` + circular mask |
| **Color correction** | White balancing | Gray World or White Patch algorithm |

## ⚠️ Common Mistakes to Avoid

1. **Gradient data types**: Always use `cv2.CV_64F`, not `cv2.CV_8U`
2. **Image paths**: Use raw strings `r'path'` or forward slashes
3. **Color channels**: OpenCV uses BGR, matplotlib uses RGB
4. **Kernel sizes**: Must be odd for most operations
5. **Image checks**: Always verify `img is not None` after loading

## 🔍 Debugging Tips

```python
# Check image properties
print(f"Shape: {img.shape}")
print(f"Type: {img.dtype}")
print(f"Min: {img.min()}, Max: {img.max()}")

# Visualize intermediate results
def show_images(imgs, titles):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i+1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title), plt.axis('off')
    plt.show()
```

## 📊 Parameter Guidelines

### Filter Sizes
- **Small (3×3, 5×5)**: Subtle effects, preserve details
- **Medium (7×7, 9×9)**: Moderate smoothing
- **Large (15×15, 21×21)**: Heavy smoothing, lose details

### Threshold Values
- **Binary threshold**: Try 127 (middle), then adjust
- **Adaptive block size**: 11, 15, 21 (must be odd)
- **Adaptive C**: Start with 2, adjust as needed

### Morphological Kernel Sizes
- **3×3**: Fine operations, preserve structure
- **5×5**: Standard operations
- **7×7+**: Coarse operations, may lose small features

## 🏆 Exam Success Strategy

1. **Read carefully**: Understand exactly what's being asked
2. **Plan approach**: Which techniques will work best?
3. **Start simple**: Get basic functionality working first
4. **Test iteratively**: Check results after each step
5. **Parameter tuning**: Adjust values based on visual results
6. **Handle errors**: Check for None images, wrong dimensions
7. **Document reasoning**: Explain why you chose specific techniques

Good luck! 🍀
