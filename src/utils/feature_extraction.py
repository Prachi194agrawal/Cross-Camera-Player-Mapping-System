"""
Feature Extraction Module for Cross-Camera Player Re-Identification
Handles visual, spatial, and temporal feature extraction for sports analytics
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Comprehensive feature extraction class for player re-identification
    Supports visual, spatial, and temporal feature extraction
    """
    
    def __init__(self, 
                 color_bins: int = 8,
                 normalize_features: bool = True,
                 feature_weights: Dict[str, float] = None):
        """
        Initialize feature extractor
        
        Args:
            color_bins (int): Number of bins for color histograms
            normalize_features (bool): Whether to normalize extracted features
            feature_weights (dict): Weights for different feature types
        """
        self.color_bins = color_bins
        self.normalize_features = normalize_features
        self.feature_weights = feature_weights or {
            'color': 1.0,
            'spatial': 1.0,
            'geometric': 0.8
        }
        
        # Feature dimensions
        self.color_feature_dims = color_bins * 3  # R, G, B histograms
        self.spatial_feature_dims = 6  # Position, size, confidence, class
        self.geometric_feature_dims = 2  # Aspect ratio, area
        self.total_visual_dims = self.color_feature_dims + self.geometric_feature_dims
        self.total_feature_dims = self.total_visual_dims + self.spatial_feature_dims
        
        print(f"🔧 FeatureExtractor initialized:")
        print(f"   Color bins: {self.color_bins}")
        print(f"   Total feature dimensions: {self.total_feature_dims}")
        print(f"   Normalization: {self.normalize_features}")

    def safe_extract_center(self, center: Union[List, Tuple, np.ndarray, float]) -> Tuple[float, float]:
        """
        Safely extract center coordinates with robust error handling
        
        Args:
            center: Center coordinate data in various formats
            
        Returns:
            Tuple[float, float]: (center_x, center_y) coordinates
        """
        try:
            if isinstance(center, (list, tuple, np.ndarray)):
                flat_center = np.array(center).flatten()
                if len(flat_center) >= 2:
                    return float(flat_center[0]), float(flat_center[1])
                elif len(flat_center) == 1:
                    return float(flat_center[0]), float(flat_center[0])
                else:
                    return 0.0, 0.0
            else:
                # Single value - use for both coordinates
                return float(center), float(center)
        except (ValueError, TypeError, IndexError):
            return 0.0, 0.0

    def extract_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract normalized color histograms for BGR channels
        
        Args:
            crop (np.ndarray): Cropped image region
            
        Returns:
            np.ndarray: Concatenated color histogram features
        """
        if crop.size == 0 or len(crop.shape) < 3:
            return np.zeros(self.color_feature_dims, dtype=np.float32)
        
        try:
            # Calculate histograms for each channel
            hist_b = cv2.calcHist([crop], [0], None, [self.color_bins], [0, 256])
            hist_g = cv2.calcHist([crop], [1], None, [self.color_bins], [0, 256])
            hist_r = cv2.calcHist([crop], [2], None, [self.color_bins], [0, 256])
            
            # Normalize histograms for illumination invariance
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
            
            # Concatenate all color features
            color_features = np.concatenate([hist_b, hist_g, hist_r])
            
            # Apply feature weights
            color_features *= self.feature_weights['color']
            
            return color_features.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Color histogram extraction failed: {e}")
            return np.zeros(self.color_feature_dims, dtype=np.float32)

    def extract_geometric_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract geometric features from cropped region
        
        Args:
            crop (np.ndarray): Cropped image region
            
        Returns:
            np.ndarray: Geometric feature vector
        """
        if crop.size == 0:
            return np.zeros(self.geometric_feature_dims, dtype=np.float32)
        
        try:
            height, width = crop.shape[:2]
            
            # Aspect ratio (width/height)
            aspect_ratio = width / (height + 1e-7)
            
            # Normalized area (scaled down for numerical stability)
            area = (width * height) / 10000.0
            
            geometric_features = np.array([aspect_ratio, area], dtype=np.float32)
            
            # Apply feature weights
            geometric_features *= self.feature_weights['geometric']
            
            return geometric_features
            
        except Exception as e:
            print(f"Warning: Geometric feature extraction failed: {e}")
            return np.zeros(self.geometric_feature_dims, dtype=np.float32)

    def extract_visual_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract comprehensive visual features from detection bounding box
        
        Args:
            frame (np.ndarray): Input video frame
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            np.ndarray: Combined visual feature vector
        """
        # Validate inputs
        if frame is None or len(bbox) < 4:
            return np.zeros(self.total_visual_dims, dtype=np.float32)
        
        # Extract and validate bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Ensure coordinates are within frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Extract crop with padding for small detections
        try:
            crop = frame[y1:y2, x1:x2]
            
            # Handle very small crops
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                return np.zeros(self.total_visual_dims, dtype=np.float32)
            
            # Extract color histogram features
            color_features = self.extract_color_histogram(crop)
            
            # Extract geometric features
            geometric_features = self.extract_geometric_features(crop)
            
            # Combine all visual features
            visual_features = np.concatenate([color_features, geometric_features])
            
            # Normalize if requested
            if self.normalize_features:
                norm = np.linalg.norm(visual_features)
                if norm > 1e-7:
                    visual_features = visual_features / norm
            
            return visual_features.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Visual feature extraction failed for bbox {bbox}: {e}")
            return np.zeros(self.total_visual_dims, dtype=np.float32)

    def extract_spatial_features(self, detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract spatial features from detection metadata
        
        Args:
            detection (dict): Detection dictionary with bbox, confidence, center, class info
            frame_shape (tuple): Frame dimensions (height, width)
            
        Returns:
            np.ndarray: Spatial feature vector
        """
        try:
            # Extract frame dimensions
            h, w = frame_shape[:2]
            
            # Extract bounding box coordinates
            bbox = detection.get('bbox', [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(coord) for coord in bbox[:4]]
            
            # Extract center coordinates safely
            center = detection.get('center', [0, 0])
            cx, cy = self.safe_extract_center(center)
            
            # Calculate normalized spatial features
            spatial_features = np.array([
                cx / (w + 1e-7),                    # Normalized center X
                cy / (h + 1e-7),                    # Normalized center Y
                (x2 - x1) / (w + 1e-7),             # Normalized width
                (y2 - y1) / (h + 1e-7),             # Normalized height
                float(detection.get('confidence', 0.0)),  # Detection confidence
                float(detection.get('class', 0))          # Class ID
            ], dtype=np.float32)
            
            # Apply feature weights
            spatial_features *= self.feature_weights['spatial']
            
            return spatial_features
            
        except Exception as e:
            print(f"Warning: Spatial feature extraction failed: {e}")
            return np.zeros(self.spatial_feature_dims, dtype=np.float32)

    def extract_comprehensive_features(self, detection: Dict[str, Any], frame: np.ndarray) -> np.ndarray:
        """
        Extract complete feature vector combining visual and spatial features
        
        Args:
            detection (dict): Detection dictionary with all metadata
            frame (np.ndarray): Video frame
            
        Returns:
            np.ndarray: Complete feature vector
        """
        try:
            # Extract visual features
            bbox = detection.get('bbox', [0, 0, 0, 0])
            visual_features = self.extract_visual_features(frame, bbox)
            
            # Extract spatial features
            spatial_features = self.extract_spatial_features(detection, frame.shape)
            
            # Combine all features
            combined_features = np.concatenate([visual_features, spatial_features])
            
            # Final normalization if requested
            if self.normalize_features:
                norm = np.linalg.norm(combined_features)
                if norm > 1e-7:
                    combined_features = combined_features / norm
            
            return combined_features.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Comprehensive feature extraction failed: {e}")
            return np.zeros(self.total_feature_dims, dtype=np.float32)

    def extract_batch_features(self, detections: List[List[Dict]], frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Extract features for all detections across multiple frames
        
        Args:
            detections (List[List[Dict]]): Nested list of detections per frame
            frames (List[np.ndarray]): List of video frames
            
        Returns:
            List[List[np.ndarray]]: Nested list of feature vectors per frame
        """
        all_features = []
        
        print(f"🔧 Extracting features for {len(frames)} frames...")
        
        for frame_idx, (frame_detections, frame) in enumerate(zip(detections, frames)):
            frame_features = []
            
            for det_idx, detection in enumerate(frame_detections):
                try:
                    # Extract comprehensive features for this detection
                    features = self.extract_comprehensive_features(detection, frame)
                    frame_features.append(features)
                    
                except Exception as e:
                    print(f"Warning: Feature extraction failed for frame {frame_idx}, detection {det_idx}: {e}")
                    # Add zero features as fallback
                    frame_features.append(np.zeros(self.total_feature_dims, dtype=np.float32))
            
            all_features.append(frame_features)
            
            # Progress feedback
            if (frame_idx + 1) % 10 == 0:
                print(f"   Processed {frame_idx + 1}/{len(frames)} frames")
        
        print(f"✅ Feature extraction complete: {sum(len(f) for f in all_features)} total feature vectors")
        return all_features

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about feature dimensions and composition
        
        Returns:
            Dict with feature information
        """
        return {
            'total_dimensions': self.total_feature_dims,
            'visual_dimensions': self.total_visual_dims,
            'spatial_dimensions': self.spatial_feature_dims,
            'color_dimensions': self.color_feature_dims,
            'geometric_dimensions': self.geometric_feature_dims,
            'color_bins': self.color_bins,
            'normalization': self.normalize_features,
            'feature_weights': self.feature_weights,
            'feature_breakdown': {
                'color_histograms': f'{self.color_feature_dims} dims (RGB histograms)',
                'geometric_features': f'{self.geometric_feature_dims} dims (aspect ratio, area)', 
                'spatial_features': f'{self.spatial_feature_dims} dims (position, size, confidence, class)'
            }
        }

class AdvancedFeatureExtractor(FeatureExtractor):
    """
    Advanced feature extractor with additional capabilities for sports analytics
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_window = 3  # Frames for temporal features
        
    def extract_temporal_features(self, detection_history: List[Dict], frame_history: List[np.ndarray]) -> np.ndarray:
        """
        Extract temporal features from detection history
        
        Args:
            detection_history: List of detections across multiple frames
            frame_history: Corresponding frames
            
        Returns:
            np.ndarray: Temporal feature vector
        """
        if len(detection_history) < 2:
            return np.zeros(4, dtype=np.float32)  # No temporal info
        
        try:
            # Calculate movement vectors
            centers = []
            for det in detection_history[-self.temporal_window:]:
                center = det.get('center', [0, 0])
                cx, cy = self.safe_extract_center(center)
                centers.append([cx, cy])
            
            if len(centers) < 2:
                return np.zeros(4, dtype=np.float32)
            
            # Calculate velocity and acceleration
            centers = np.array(centers)
            velocities = np.diff(centers, axis=0)
            
            if len(velocities) > 0:
                avg_velocity = np.mean(velocities, axis=0)
                velocity_magnitude = np.linalg.norm(avg_velocity)
                
                if len(velocities) > 1:
                    accelerations = np.diff(velocities, axis=0)
                    avg_acceleration = np.mean(accelerations, axis=0)
                    acceleration_magnitude = np.linalg.norm(avg_acceleration)
                else:
                    acceleration_magnitude = 0.0
            else:
                velocity_magnitude = 0.0
                acceleration_magnitude = 0.0
            
            temporal_features = np.array([
                velocity_magnitude,
                acceleration_magnitude,
                len(detection_history),  # Track length
                1.0 if len(detection_history) > 5 else 0.0  # Stability indicator
            ], dtype=np.float32)
            
            return temporal_features
            
        except Exception as e:
            print(f"Warning: Temporal feature extraction failed: {e}")
            return np.zeros(4, dtype=np.float32)

    def extract_team_context_features(self, detection: Dict, nearby_detections: List[Dict]) -> np.ndarray:
        """
        Extract team/context features based on nearby detections
        
        Args:
            detection: Target detection
            nearby_detections: Other detections in the same frame
            
        Returns:
            np.ndarray: Context feature vector
        """
        try:
            # Count nearby objects by class
            class_counts = {'player': 0, 'goalkeeper': 0, 'referee': 0, 'ball': 0}
            
            target_center = detection.get('center', [0, 0])
            tx, ty = self.safe_extract_center(target_center)
            
            for other_det in nearby_detections:
                other_center = other_det.get('center', [0, 0])
                ox, oy = self.safe_extract_center(other_center)
                
                # Calculate distance
                distance = np.sqrt((tx - ox)**2 + (ty - oy)**2)
                
                # Only consider nearby objects (within reasonable distance)
                if distance < 200:  # Pixel distance threshold
                    class_name = other_det.get('class_name', 'unknown')
                    if class_name in class_counts:
                        class_counts[class_name] += 1
            
            context_features = np.array([
                class_counts['player'],
                class_counts['goalkeeper'], 
                class_counts['referee'],
                class_counts['ball'],
                len(nearby_detections),  # Total nearby objects
                1.0 if class_counts['ball'] > 0 else 0.0  # Ball presence
            ], dtype=np.float32)
            
            return context_features
            
        except Exception as e:
            print(f"Warning: Context feature extraction failed: {e}")
            return np.zeros(6, dtype=np.float32)

# Utility functions for feature processing
def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize feature vector to unit length"""
    norm = np.linalg.norm(features)
    if norm > 1e-7:
        return features / norm
    return features

def combine_feature_vectors(feature_lists: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """Combine multiple feature vectors with optional weights"""
    if not feature_lists:
        return np.array([])
    
    if weights is None:
        weights = [1.0] * len(feature_lists)
    
    # Apply weights and concatenate
    weighted_features = []
    for features, weight in zip(feature_lists, weights):
        weighted_features.append(features * weight)
    
    return np.concatenate(weighted_features)

def feature_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Calculate cosine similarity between two feature vectors"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        return float(cosine_similarity([feat1], [feat2])[0, 0])
    except:
        # Fallback implementation
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 > 1e-7 and norm2 > 1e-7:
            return float(np.dot(feat1, feat2) / (norm1 * norm2))
        return 0.0

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Feature Extraction Module...")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(color_bins=8, normalize_features=True)
    
    # Print feature information
    info = extractor.get_feature_info()
    print(f"Feature extractor info: {info}")
    
    # Create dummy data for testing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_detection = {
        'bbox': [100, 100, 200, 200],
        'confidence': 0.85,
        'center': [150, 150],
        'class': 2,
        'class_name': 'player'
    }
    
    # Test feature extraction
    features = extractor.extract_comprehensive_features(dummy_detection, dummy_frame)
    print(f"✅ Extracted features: {features.shape} dimensions")
    print(f"Feature vector preview: {features[:10]}...")
    
    print("🎉 Feature extraction module test completed!")
