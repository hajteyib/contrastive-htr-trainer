import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform as skimage_transform
import torchvision.transforms as transforms


class IntelligentBackgroundNormalizer:
    """Intelligent background detection and normalization for HTR."""
    
    def __init__(self, target_bg_value: float = 1.0):
        self.target_bg_value = target_bg_value
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Normalize background to white while preserving ink contrast."""
        # Detect background (most common pixel value in corners)
        h, w = image.shape[:2]
        corner_regions = [
            image[:h//4, :w//4],  # Top-left
            image[:h//4, -w//4:], # Top-right  
            image[-h//4:, :w//4], # Bottom-left
            image[-h//4:, -w//4:] # Bottom-right
        ]
        
        # Find most common value in corners (background)
        corner_pixels = np.concatenate([region.flatten() for region in corner_regions])
        hist, bins = np.histogram(corner_pixels, bins=50)
        bg_value = bins[np.argmax(hist)]
        
        # Normalize: background -> target_bg_value, preserve relative contrasts
        if abs(bg_value - self.target_bg_value) > 0.1:
            # Simple linear normalization
            if bg_value > 0.5:  # Light background
                normalized = image / bg_value * self.target_bg_value
            else:  # Dark background
                normalized = (1 - image) / (1 - bg_value) * self.target_bg_value
                
            return np.clip(normalized, 0, 1)
        
        return image


class ElasticDistortion:
    """Enhanced elastic distortion for natural handwriting variation."""
    
    def __init__(self, alpha: float = 15, sigma: float = 4, probability: float = 0.4):
        self.alpha = alpha
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic distortion with probability."""
        if random.random() > self.probability:
            return image
            
        height, width = image.shape[:2]
        
        # Generate smooth random displacement fields
        dx = gaussian_filter(
            (np.random.rand(height, width) * 2 - 1), 
            self.sigma, mode="constant", cval=0
        ) * self.alpha
        
        dy = gaussian_filter(
            (np.random.rand(height, width) * 2 - 1), 
            self.sigma, mode="constant", cval=0
        ) * self.alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply distortion
        distorted = map_coordinates(
            image, indices, order=1, mode='reflect'
        ).reshape((height, width))
        
        return distorted


class InkVariationSimulator:
    """Simulate realistic ink thickness and intensity variations."""
    
    def __init__(self, intensity_range: Tuple[float, float] = (0.8, 1.2),
                 thickness_prob: float = 0.3, probability: float = 0.5):
        self.intensity_range = intensity_range
        self.thickness_prob = thickness_prob
        self.probability = probability
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply ink variations."""
        if random.random() > self.probability:
            return image
            
        result = image.copy()
        
        # Intensity variation (simulate ink density changes)
        intensity_factor = random.uniform(*self.intensity_range)
        
        # Only affect darker regions (ink regions)
        ink_mask = image < 0.7  # Assume darker regions are ink
        result[ink_mask] = np.clip(result[ink_mask] * intensity_factor, 0, 1)
        
        # Thickness variation via morphological operations
        if random.random() < self.thickness_prob:
            kernel_size = random.choice([2, 3])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            if random.random() < 0.5:
                # Thicker strokes (dilation)
                result = cv2.dilate(result, kernel, iterations=1)
            else:
                # Thinner strokes (erosion) 
                result = cv2.erode(result, kernel, iterations=1)
        
        return np.clip(result, 0, 1)


class PaperAgeingSimulator:
    """Simulate paper aging and texture effects."""
    
    def __init__(self, noise_intensity: float = 0.015, 
                 stain_prob: float = 0.15, probability: float = 0.4):
        self.noise_intensity = noise_intensity
        self.stain_prob = stain_prob
        self.probability = probability
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply paper aging effects."""
        if random.random() > self.probability:
            return image
            
        height, width = image.shape[:2]
        result = image.copy()
        
        # Paper texture noise
        paper_noise = np.random.normal(0, self.noise_intensity, (height, width))
        result = result + paper_noise
        
        # Random stains/spots
        if random.random() < self.stain_prob:
            num_stains = random.randint(1, 3)
            for _ in range(num_stains):
                # Random stain parameters
                center_x = random.randint(width//4, 3*width//4)
                center_y = random.randint(height//4, 3*height//4)
                radius = random.randint(10, 25)
                intensity = random.uniform(0.05, 0.2)
                
                # Create circular stain with Gaussian falloff
                y, x = np.ogrid[:height, :width]
                distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                stain_mask = distances <= radius
                
                # Apply Gaussian falloff
                gaussian_falloff = np.exp(-(distances**2) / (2 * (radius/2)**2))
                stain_effect = intensity * gaussian_falloff
                
                # Add stain (darken background slightly)
                result[stain_mask] -= stain_effect[stain_mask]
        
        return np.clip(result, 0, 1)


class StructuralHTRAugmentations:
    """HTR-specific structural augmentations."""
    
    def __init__(self, word_mask_prob: float = 0.2, 
                 line_dropout_prob: float = 0.1,
                 sliding_window_prob: float = 0.15):
        self.word_mask_prob = word_mask_prob
        self.line_dropout_prob = line_dropout_prob
        self.sliding_window_prob = sliding_window_prob
    
    def _word_level_masking(self, image: np.ndarray, max_masks: int = 4) -> np.ndarray:
        """Mask random word-sized regions."""
        height, width = image.shape[:2]
        result = image.copy()
        
        num_masks = random.randint(1, max_masks)
        
        for _ in range(num_masks):
            # Word-sized mask dimensions
            mask_width = random.randint(30, 100)  # Typical word width
            mask_height = random.randint(20, 40)   # Typical word height
            
            # Random position
            start_x = random.randint(0, max(1, width - mask_width))
            start_y = random.randint(0, max(1, height - mask_height))
            
            # Apply mask (set to background color)
            result[start_y:start_y + mask_height, 
                   start_x:start_x + mask_width] = 1.0  # White background
        
        return result
    
    def _sliding_window_crop(self, image: np.ndarray) -> np.ndarray:
        """Apply horizontal sliding window crop."""
        height, width = image.shape[:2]
        
        # Random crop parameters
        min_width_ratio = 0.6
        crop_width = random.randint(int(width * min_width_ratio), width)
        start_x = random.randint(0, width - crop_width)
        
        # Extract and resize
        cropped = image[:, start_x:start_x + crop_width]
        resized = cv2.resize(cropped, (width, height))
        
        return resized
    
    def _line_dropout(self, image: np.ndarray) -> np.ndarray:
        """Randomly dropout horizontal bands (simulate missing lines)."""
        height, width = image.shape[:2]
        result = image.copy()
        
        # Estimate line height
        estimated_line_height = max(15, height // 20)  # Conservative estimate
        num_possible_lines = height // estimated_line_height
        
        if num_possible_lines > 3:  # Only if we have multiple lines
            num_dropouts = random.randint(1, min(2, num_possible_lines // 3))
            
            for _ in range(num_dropouts):
                # Random line to dropout
                line_idx = random.randint(0, num_possible_lines - 1)
                start_y = line_idx * estimated_line_height
                end_y = min(start_y + estimated_line_height, height)
                
                # Set line region to background
                result[start_y:end_y, :] = 1.0
        
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply structural augmentations with probability."""
        # Choose one structural augmentation
        rand_val = random.random()
        
        if rand_val < self.word_mask_prob:
            return self._word_level_masking(image)
        elif rand_val < self.word_mask_prob + self.sliding_window_prob:
            return self._sliding_window_crop(image)
        elif rand_val < self.word_mask_prob + self.sliding_window_prob + self.line_dropout_prob:
            return self._line_dropout(image)
        else:
            return image  # No structural augmentation


class OptimizedHTRAugmentation:
    """Complete optimized HTR augmentation pipeline."""
    
    def __init__(self, 
                 geometric_prob: float = 0.4,
                 photometric_prob: float = 0.5, 
                 structural_prob: float = 0.25,
                 preserve_quality: bool = True):
        
        # Background normalization (always applied)
        self.bg_normalizer = IntelligentBackgroundNormalizer()
        
        # Geometric augmentations
        self.elastic = ElasticDistortion(alpha=15, sigma=4, probability=0.6)
        self.geometric_prob = geometric_prob
        
        # Photometric augmentations  
        self.ink_variation = InkVariationSimulator(probability=0.7)
        self.paper_aging = PaperAgeingSimulator(probability=0.6)
        self.photometric_prob = photometric_prob
        
        # Structural augmentations
        self.structural_augs = StructuralHTRAugmentations()
        self.structural_prob = structural_prob
        
        # Quality preservation
        self.preserve_quality = preserve_quality
        
        # Standard geometric transforms
        self.max_rotation = 2.0  # Reduced for better readability
        self.max_shear = 0.05
        self.perspective_strength = 0.0002
    
    def _apply_geometric_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply geometric transformations."""
        if random.random() > self.geometric_prob:
            return image
        
        result = image.copy()
        
        # Elastic distortion (most important for HTR)
        result = self.elastic(result)
        
        # Standard geometric transforms with conservative parameters
        if random.random() < 0.3:  # Rotation
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            center = (result.shape[1] // 2, result.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, rotation_matrix, 
                                   (result.shape[1], result.shape[0]), 
                                   borderValue=1.0)
        
        if random.random() < 0.2:  # Shear
            shear_x = random.uniform(-self.max_shear, self.max_shear)
            shear_matrix = np.array([[1, shear_x, 0], [0, 1, 0]], dtype=np.float32)
            result = cv2.warpAffine(result, shear_matrix,
                                   (result.shape[1], result.shape[0]),
                                   borderValue=1.0)
        
        if random.random() < 0.15:  # Subtle perspective
            h, w = result.shape[:2]
            # Very subtle perspective distortion
            delta = int(w * self.perspective_strength)
            src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
            dst_points = np.float32([
                [random.randint(-delta, delta), random.randint(-delta, delta)],
                [w-1+random.randint(-delta, delta), random.randint(-delta, delta)],
                [random.randint(-delta, delta), h-1+random.randint(-delta, delta)],
                [w-1+random.randint(-delta, delta), h-1+random.randint(-delta, delta)]
            ])
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            result = cv2.warpPerspective(result, perspective_matrix, (w, h), borderValue=1.0)
        
        return result
    
    def _apply_photometric_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric transformations."""
        if random.random() > self.photometric_prob:
            return image
        
        result = image.copy()
        
        # HTR-specific photometric augmentations
        result = self.ink_variation(result)
        result = self.paper_aging(result)
        
        # Standard photometric adjustments
        if random.random() < 0.4:  # Contrast adjustment
            contrast_factor = random.uniform(0.85, 1.15)
            result = np.clip((result - 0.5) * contrast_factor + 0.5, 0, 1)
        
        if random.random() < 0.3:  # Brightness adjustment
            brightness_delta = random.uniform(-0.05, 0.05)
            result = np.clip(result + brightness_delta, 0, 1)
        
        return result
    
    def _apply_structural_transforms(self, image: np.ndarray) -> np.ndarray:
        """Apply structural transformations."""
        if random.random() > self.structural_prob:
            return image
        
        return self.structural_augs(image)
    
    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]: # On change le type de retour
        """
        Apply complete HTR augmentation pipeline.
        
        Args:
            image: Input image as numpy array [H, W] or [H, W, C]
            
        Returns:
            A dictionary containing the augmented image under the key 'image'.
        """
        # ... (tout le début de la fonction ne change pas)
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        
        result = self.bg_normalizer(image)
        result = self._apply_geometric_transforms(result)
        result = self._apply_photometric_transforms(result)
        result = self._apply_structural_transforms(result)
        
        if self.preserve_quality:
            if result.std() < 0.05:
                result = np.clip((result - result.mean()) * 2 + result.mean(), 0, 1)
            result = np.clip(result, 0, 1)
        
        ### CORRECTION ###
        # On retourne un dictionnaire, comme attendu.
        return {'image': result}


class SmartResize:
    """Enhanced smart resizing for variable HTR images."""
    
    def __init__(self, target_height: int = 512, pad_value: float = 1.0,
                 preserve_aspect_ratio: bool = True):
        self.target_height = target_height
        self.pad_value = pad_value
        self.preserve_aspect_ratio = preserve_aspect_ratio
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Smart resize with aspect ratio preservation and efficient padding.
        
        Args:
            image: Input image [H, W] or [H, W, C]
            
        Returns:
            Resized and padded image, metadata dict
        """
        original_height, original_width = image.shape[:2]
        
        if self.preserve_aspect_ratio:
            # Calculate new dimensions preserving aspect ratio
            scale_factor = self.target_height / original_height
            new_width = int(original_width * scale_factor)
        else:
            # Fixed aspect ratio resize
            scale_factor = self.target_height / original_height
            new_width = int(original_width * scale_factor)
        
        # Resize image
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_width, self.target_height), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_width, self.target_height), interpolation=cv2.INTER_AREA)
        
        # Smart padding - pad to multiple of 32 for efficient processing
        padded_width = ((new_width + 31) // 32) * 32
        pad_amount = padded_width - new_width
        
        if pad_amount > 0:
            if len(resized.shape) == 3:
                padding = ((0, 0), (0, pad_amount), (0, 0))
            else:
                padding = ((0, 0), (0, pad_amount))
            
            resized = np.pad(resized, padding, mode='constant', constant_values=self.pad_value)
        
        metadata = {
            'original_size': (original_height, original_width),
            'scale_factor': scale_factor,
            'new_size': (self.target_height, new_width),
            'padded_width': padded_width,
            'pad_amount': pad_amount
        }
        
        return resized, metadata


def test_optimized_augmentations():
    """Test the optimized augmentation pipeline."""
    print("Testing optimized HTR augmentations...")
    
    # Create sample handwriting-like image
    image = np.ones((400, 1000), dtype=np.float32)  # White background
    
    # Add some "handwriting" strokes with noise
    for i in range(12):  # 12 lines
        y = 30 + i * 30
        for j in range(0, 900, 80):  # Words
            x_start, x_end = j + 10, j + 60
            if x_end < 1000 and y < 400:
                # Add character strokes
                image[y-3:y+3, x_start:x_end] = np.random.uniform(0.1, 0.4)
                
                # Add some connecting strokes
                if j > 0:
                    image[y-1:y+1, x_start-10:x_start] = np.random.uniform(0.3, 0.6)
    
    # Add some background texture
    noise = np.random.normal(0.95, 0.02, image.shape)
    image = np.clip(image + (noise - image) * 0.1, 0, 1)
    
    # Test resizer
    resizer = SmartResize(target_height=512)
    resized, metadata = resizer(image)
    print(f"Resizing: {image.shape} -> {resized.shape}")
    print(f"Metadata: {metadata}")
    
    # Test augmentation pipeline
    augmenter = OptimizedHTRAugmentation(
        geometric_prob=0.6,
        photometric_prob=0.7,
        structural_prob=0.4
    )
    
    print("\nTesting augmentation pipeline:")
    for i in range(5):
        augmented = augmenter(resized.copy())
        print(f"  Aug {i+1}: shape {augmented.shape}, "
              f"range [{augmented.min():.3f}, {augmented.max():.3f}], "
              f"std {augmented.std():.3f}")
    
    print("✅ Optimized augmentation tests passed!")


if __name__ == "__main__":
    test_optimized_augmentations() 


