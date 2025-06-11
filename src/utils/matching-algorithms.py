"""
Matching Algorithms Module for Cross-Camera Player Re-Identification
Implements various matching strategies for sports analytics
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import warnings
warnings.filterwarnings('ignore')

class MatchingAlgorithms:
    """
    Comprehensive matching algorithms for cross-camera object re-identification
    Supports multiple matching strategies and similarity metrics
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.02,
                 class_boost_factor: float = 1.5,
                 temporal_weight: float = 0.3,
                 distance_weight: float = 0.2):
        """
        Initialize matching algorithms with configuration parameters
        
        Args:
            similarity_threshold (float): Minimum similarity for valid matches
            class_boost_factor (float): Multiplier for same-class matches
            temporal_weight (float): Weight for temporal consistency
            distance_weight (float): Weight for spatial distance
        """
        self.similarity_threshold = similarity_threshold
        self.class_boost_factor = class_boost_factor
        self.temporal_weight = temporal_weight
        self.distance_weight = distance_weight
        
        print(f"🔧 MatchingAlgorithms initialized:")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        print(f"   Class boost factor: {self.class_boost_factor}")
        print(f"   Temporal weight: {self.temporal_weight}")

    def compute_cosine_similarity_matrix(self, 
                                       features1: List[np.ndarray], 
                                       features2: List[np.ndarray]) -> np.ndarray:
        """
        Compute cosine similarity matrix between two sets of feature vectors
        Based on search results[5] methodology
        
        Args:
            features1: Feature vectors from camera 1
            features2: Feature vectors from camera 2
            
        Returns:
            np.ndarray: Similarity matrix of shape (len(features1), len(features2))
        """
        if len(features1) == 0 or len(features2) == 0:
            return np.array([])
        
        try:
            # Convert to numpy arrays
            feat1_array = np.array(features1)
            feat2_array = np.array(features2)
            
            # Validate feature dimensions
            if feat1_array.ndim != 2 or feat2_array.ndim != 2:
                feat1_array = feat1_array.reshape(len(features1), -1)
                feat2_array = feat2_array.reshape(len(features2), -1)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(feat1_array, feat2_array)
            
            return similarity_matrix.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Cosine similarity computation failed: {e}")
            return np.zeros((len(features1), len(features2)), dtype=np.float32)

    def compute_euclidean_similarity_matrix(self, 
                                          features1: List[np.ndarray], 
                                          features2: List[np.ndarray]) -> np.ndarray:
        """
        Compute Euclidean distance-based similarity matrix
        
        Args:
            features1: Feature vectors from camera 1
            features2: Feature vectors from camera 2
            
        Returns:
            np.ndarray: Similarity matrix (converted from distances)
        """
        if len(features1) == 0 or len(features2) == 0:
            return np.array([])
        
        try:
            feat1_array = np.array(features1)
            feat2_array = np.array(features2)
            
            # Validate feature dimensions
            if feat1_array.ndim != 2 or feat2_array.ndim != 2:
                feat1_array = feat1_array.reshape(len(features1), -1)
                feat2_array = feat2_array.reshape(len(features2), -1)
            
            # Compute Euclidean distances
            distance_matrix = euclidean_distances(feat1_array, feat2_array)
            
            # Convert distances to similarities (higher distance = lower similarity)
            max_distance = np.max(distance_matrix) + 1e-7
            similarity_matrix = 1.0 - (distance_matrix / max_distance)
            
            return similarity_matrix.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Euclidean similarity computation failed: {e}")
            return np.zeros((len(features1), len(features2)), dtype=np.float32)

    def apply_class_aware_boosting(self, 
                                 similarity_matrix: np.ndarray,
                                 features1: List[np.ndarray], 
                                 features2: List[np.ndarray]) -> np.ndarray:
        """
        Apply class-aware boosting to enhance same-class similarities
        Based on the original pipeline implementation
        
        Args:
            similarity_matrix: Base similarity matrix
            features1: Feature vectors from camera 1 (with class info)
            features2: Feature vectors from camera 2 (with class info)
            
        Returns:
            np.ndarray: Boosted similarity matrix
        """
        if similarity_matrix.size == 0:
            return similarity_matrix
        
        try:
            boosted_matrix = similarity_matrix.copy()
            
            # Extract class information (assumes class is the last feature)
            for i, f1 in enumerate(features1):
                for j, f2 in enumerate(features2):
                    if len(f1) > 0 and len(f2) > 0:
                        class1 = f1[-1] if len(f1) > 1 else 0
                        class2 = f2[-1] if len(f2) > 1 else 0
                        
                        # Boost similarity for same class
                        if abs(class1 - class2) < 1e-7:  # Same class
                            boosted_matrix[i, j] *= self.class_boost_factor
            
            return boosted_matrix
            
        except Exception as e:
            print(f"Warning: Class-aware boosting failed: {e}")
            return similarity_matrix

    def compute_spatial_distance_penalty(self, 
                                       detections1: List[Dict],
                                       detections2: List[Dict],
                                       frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Compute spatial distance penalty matrix for realistic matching
        
        Args:
            detections1: Detections from camera 1 with center coordinates
            detections2: Detections from camera 2 with center coordinates
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            np.ndarray: Distance penalty matrix (lower is better)
        """
        if len(detections1) == 0 or len(detections2) == 0:
            return np.array([])
        
        try:
            h, w = frame_shape[:2]
            penalty_matrix = np.ones((len(detections1), len(detections2)), dtype=np.float32)
            
            for i, det1 in enumerate(detections1):
                center1 = det1.get('center', [0, 0])
                cx1, cy1 = self._safe_extract_center(center1)
                
                for j, det2 in enumerate(detections2):
                    center2 = det2.get('center', [0, 0])
                    cx2, cy2 = self._safe_extract_center(center2)
                    
                    # Normalize coordinates
                    nx1, ny1 = cx1 / (w + 1e-7), cy1 / (h + 1e-7)
                    nx2, ny2 = cx2 / (w + 1e-7), cy2 / (h + 1e-7)
                    
                    # Calculate normalized distance
                    distance = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2)
                    
                    # Convert to penalty (higher distance = higher penalty)
                    penalty_matrix[i, j] = distance
            
            # Normalize penalties to [0, 1] range
            max_penalty = np.max(penalty_matrix) + 1e-7
            penalty_matrix = penalty_matrix / max_penalty
            
            return penalty_matrix
            
        except Exception as e:
            print(f"Warning: Spatial distance penalty computation failed: {e}")
            return np.ones((len(detections1), len(detections2)), dtype=np.float32)

    def hungarian_matching(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Perform optimal matching using Hungarian algorithm
        Based on search results[4] and [6] methodology
        
        Args:
            similarity_matrix: Similarity matrix between objects
            
        Returns:
            List of matches as (idx1, idx2, similarity_score) tuples
        """
        if similarity_matrix.size == 0:
            return []
        
        try:
            # Convert similarity to cost matrix (Hungarian finds minimum cost)
            cost_matrix = -similarity_matrix
            
            # Apply Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Filter matches by similarity threshold
            matches = []
            for row, col in zip(row_indices, col_indices):
                similarity_score = similarity_matrix[row, col]
                if similarity_score > self.similarity_threshold:
                    matches.append((int(row), int(col), float(similarity_score)))
            
            return matches
            
        except Exception as e:
            print(f"Warning: Hungarian matching failed: {e}")
            return []

    def greedy_matching(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Perform greedy matching as alternative to Hungarian algorithm
        
        Args:
            similarity_matrix: Similarity matrix between objects
            
        Returns:
            List of matches as (idx1, idx2, similarity_score) tuples
        """
        if similarity_matrix.size == 0:
            return []
        
        try:
            matches = []
            used_rows = set()
            used_cols = set()
            
            # Create list of all similarities with indices
            similarities = []
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    if similarity_matrix[i, j] > self.similarity_threshold:
                        similarities.append((similarity_matrix[i, j], i, j))
            
            # Sort by similarity (highest first)
            similarities.sort(reverse=True)
            
            # Greedily assign matches
            for similarity, i, j in similarities:
                if i not in used_rows and j not in used_cols:
                    matches.append((i, j, similarity))
                    used_rows.add(i)
                    used_cols.add(j)
            
            return matches
            
        except Exception as e:
            print(f"Warning: Greedy matching failed: {e}")
            return []

    def bipartite_matching(self, 
                          similarity_matrix: np.ndarray,
                          method: str = 'hungarian') -> List[Tuple[int, int, float]]:
        """
        General bipartite matching with multiple algorithm options
        
        Args:
            similarity_matrix: Similarity matrix between objects
            method: Matching method ('hungarian', 'greedy')
            
        Returns:
            List of matches as (idx1, idx2, similarity_score) tuples
        """
        if method.lower() == 'hungarian':
            return self.hungarian_matching(similarity_matrix)
        elif method.lower() == 'greedy':
            return self.greedy_matching(similarity_matrix)
        else:
            print(f"Warning: Unknown matching method '{method}', using Hungarian")
            return self.hungarian_matching(similarity_matrix)

    def temporal_consistency_matching(self, 
                                    current_similarity: np.ndarray,
                                    previous_matches: List[Tuple[int, int, float]],
                                    detections1: List[Dict],
                                    detections2: List[Dict]) -> List[Tuple[int, int, float]]:
        """
        Perform matching with temporal consistency from previous frame
        
        Args:
            current_similarity: Current frame similarity matrix
            previous_matches: Matches from previous frame
            detections1: Current detections from camera 1
            detections2: Current detections from camera 2
            
        Returns:
            List of temporally consistent matches
        """
        if current_similarity.size == 0:
            return []
        
        try:
            # Start with current similarity
            enhanced_similarity = current_similarity.copy()
            
            # Apply temporal consistency boost
            for prev_i, prev_j, prev_score in previous_matches:
                # Find closest current detections to previous ones
                best_i, best_j = self._find_closest_detections(
                    prev_i, prev_j, detections1, detections2
                )
                
                if (best_i < enhanced_similarity.shape[0] and 
                    best_j < enhanced_similarity.shape[1]):
                    # Boost similarity for temporally consistent matches
                    enhanced_similarity[best_i, best_j] += self.temporal_weight * prev_score
            
            # Apply Hungarian matching on enhanced similarity
            return self.hungarian_matching(enhanced_similarity)
            
        except Exception as e:
            print(f"Warning: Temporal consistency matching failed: {e}")
            return self.hungarian_matching(current_similarity)

    def ensemble_matching(self, 
                         features1: List[np.ndarray],
                         features2: List[np.ndarray],
                         detections1: List[Dict] = None,
                         detections2: List[Dict] = None,
                         frame_shape: Tuple[int, int] = None) -> List[Tuple[int, int, float]]:
        """
        Ensemble matching combining multiple similarity metrics
        
        Args:
            features1: Feature vectors from camera 1
            features2: Feature vectors from camera 2
            detections1: Optional detection metadata for spatial analysis
            detections2: Optional detection metadata for spatial analysis
            frame_shape: Optional frame shape for spatial normalization
            
        Returns:
            List of matches from ensemble approach
        """
        if len(features1) == 0 or len(features2) == 0:
            return []
        
        try:
            # Compute multiple similarity matrices
            cosine_sim = self.compute_cosine_similarity_matrix(features1, features2)
            euclidean_sim = self.compute_euclidean_similarity_matrix(features1, features2)
            
            # Combine similarities with weights
            ensemble_similarity = 0.6 * cosine_sim + 0.4 * euclidean_sim
            
            # Apply class-aware boosting
            ensemble_similarity = self.apply_class_aware_boosting(
                ensemble_similarity, features1, features2
            )
            
            # Apply spatial distance penalty if spatial info available
            if (detections1 is not None and detections2 is not None and 
                frame_shape is not None):
                spatial_penalty = self.compute_spatial_distance_penalty(
                    detections1, detections2, frame_shape
                )
                if spatial_penalty.size > 0:
                    ensemble_similarity -= self.distance_weight * spatial_penalty
            
            # Perform Hungarian matching
            return self.hungarian_matching(ensemble_similarity)
            
        except Exception as e:
            print(f"Warning: Ensemble matching failed: {e}")
            return self.hungarian_matching(
                self.compute_cosine_similarity_matrix(features1, features2)
            )

    def multi_frame_matching(self, 
                           features_sequence1: List[List[np.ndarray]],
                           features_sequence2: List[List[np.ndarray]],
                           method: str = 'hungarian') -> List[List[Tuple[int, int, float]]]:
        """
        Perform matching across multiple frames with temporal consistency
        
        Args:
            features_sequence1: Sequence of feature lists from camera 1
            features_sequence2: Sequence of feature lists from camera 2
            method: Matching method to use
            
        Returns:
            List of match lists for each frame
        """
        all_matches = []
        previous_matches = []
        
        print(f"🔗 Performing multi-frame matching across {len(features_sequence1)} frames...")
        
        for frame_idx, (feat1, feat2) in enumerate(zip(features_sequence1, features_sequence2)):
            # Compute current frame similarity
            similarity_matrix = self.compute_cosine_similarity_matrix(feat1, feat2)
            
            # Apply class-aware boosting
            similarity_matrix = self.apply_class_aware_boosting(similarity_matrix, feat1, feat2)
            
            # Use temporal consistency if available
            if previous_matches and self.temporal_weight > 0:
                # Note: This would need detection metadata for full implementation
                current_matches = self.hungarian_matching(similarity_matrix)
            else:
                current_matches = self.bipartite_matching(similarity_matrix, method)
            
            all_matches.append(current_matches)
            previous_matches = current_matches
            
            if (frame_idx + 1) % 10 == 0:
                print(f"   Processed {frame_idx + 1}/{len(features_sequence1)} frames")
        
        print(f"✅ Multi-frame matching complete")
        return all_matches

    def _safe_extract_center(self, center: Union[List, Tuple, np.ndarray, float]) -> Tuple[float, float]:
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
                return float(center), float(center)
        except (ValueError, TypeError, IndexError):
            return 0.0, 0.0

    def _find_closest_detections(self, 
                               prev_i: int, prev_j: int,
                               current_detections1: List[Dict],
                               current_detections2: List[Dict]) -> Tuple[int, int]:
        """
        Find closest current detections to previous frame detections
        
        Args:
            prev_i: Previous detection index in camera 1
            prev_j: Previous detection index in camera 2
            current_detections1: Current detections from camera 1
            current_detections2: Current detections from camera 2
            
        Returns:
            Tuple of closest detection indices
        """
        # Simplified implementation - returns first valid indices
        # In practice, this would compare spatial positions
        best_i = min(prev_i, len(current_detections1) - 1) if current_detections1 else 0
        best_j = min(prev_j, len(current_detections2) - 1) if current_detections2 else 0
        return max(0, best_i), max(0, best_j)

    def get_matching_statistics(self, matches: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """
        Compute statistics for matching results
        
        Args:
            matches: List of matches with similarity scores
            
        Returns:
            Dictionary with matching statistics
        """
        if not matches:
            return {
                'total_matches': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'similarity_std': 0.0
            }
        
        similarities = [match[2] for match in matches]
        
        return {
            'total_matches': len(matches),
            'avg_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'similarity_std': float(np.std(similarities))
        }

# Utility functions for advanced matching scenarios
def compute_iou_similarity(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) similarity between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU similarity score
    """
    try:
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-7)
        
    except Exception:
        return 0.0

def apply_non_maximum_suppression(matches: List[Tuple[int, int, float]], 
                                 threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping matches
    
    Args:
        matches: List of matches with similarity scores
        threshold: NMS threshold
        
    Returns:
        Filtered list of matches
    """
    if not matches:
        return []
    
    # Sort matches by similarity score (descending)
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
    
    # Track used indices
    used_indices1 = set()
    used_indices2 = set()
    
    filtered_matches = []
    for i1, i2, score in sorted_matches:
        if i1 not in used_indices1 and i2 not in used_indices2:
            filtered_matches.append((i1, i2, score))
            used_indices1.add(i1)
            used_indices2.add(i2)
    
    return filtered_matches

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Matching Algorithms Module...")
    
    # Initialize matching algorithms
    matcher = MatchingAlgorithms(
        similarity_threshold=0.02,
        class_boost_factor=1.5,
        temporal_weight=0.3
    )
    
    # Create dummy feature data for testing
    features1 = [np.random.rand(32) for _ in range(5)]
    features2 = [np.random.rand(32) for _ in range(7)]
    
    # Test cosine similarity
    cosine_sim = matcher.compute_cosine_similarity_matrix(features1, features2)
    print(f"✅ Cosine similarity matrix shape: {cosine_sim.shape}")
    
    # Test Hungarian matching
    matches = matcher.hungarian_matching(cosine_sim)
    print(f"✅ Hungarian matches found: {len(matches)}")
    
    # Test ensemble matching
    ensemble_matches = matcher.ensemble_matching(features1, features2)
    print(f"✅ Ensemble matches found: {len(ensemble_matches)}")
    
    # Get statistics
    stats = matcher.get_matching_statistics(matches)
    print(f"✅ Matching statistics: {stats}")
    
    print("🎉 Matching algorithms module test completed!")
