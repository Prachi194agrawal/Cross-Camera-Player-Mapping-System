# API Reference

## Overview

This document provides comprehensive API documentation for the Cross-Camera Player Mapping System. The system is built around the `PlayerReIDPipeline` class and supporting functions that enable robust player re-identification across different camera viewpoints.

## Core Classes

### PlayerReIDPipeline

Main pipeline class for cross-camera player re-identification and object detection.

```python
class PlayerReIDPipeline:
    """
    Comprehensive pipeline for cross-camera object detection and matching.
    
    Attributes:
        model (YOLO): Loaded YOLOv8 detection model
        broadcast_detections (list): Detection results from broadcast camera
        tacticam_detections (list): Detection results from tactical camera  
        broadcast_frames (list): Extracted frames from broadcast video
        tacticam_frames (list): Extracted frames from tactical video
    """
```

#### Constructor

```python
def __init__(self, model_path: str)
```

Initialize the pipeline with a pre-trained YOLO model.

**Parameters:**
- `model_path` (str): Path to the YOLOv8 model file (.pt format)

**Raises:**
- `FileNotFoundError`: If model file doesn't exist
- `RuntimeError`: If model cannot be loaded

**Example:**
```python
pipeline = PlayerReIDPipeline('/path/to/best.pt')
```

---

## Core Methods

### Frame Processing

#### extract_frames

```python
def extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]
```

Extract strategic frame samples from video for processing.

**Parameters:**
- `video_path` (str): Path to input video file
- `max_frames` (int, optional): Maximum number of frames to extract. Default: 30

**Returns:**
- `List[np.ndarray]`: List of extracted frame images

**Implementation Details:**
- Samples from middle 50% of video content (25%-75% range)
- Avoids scene transitions and unstable footage
- Uses intelligent step calculation for uniform sampling

**Example:**
```python
frames = pipeline.extract_frames('/path/to/video.mp4', max_frames=30)
print(f"Extracted {len(frames)} frames")
```

---

### Object Detection

#### detect_all_objects

```python
def detect_all_objects(self, frames: List[np.ndarray], video_name: str = "") -> List[List[Dict]]
```

Detect all objects in video frames using YOLO model.

**Parameters:**
- `frames` (List[np.ndarray]): List of video frames
- `video_name` (str, optional): Video identifier for logging

**Returns:**
- `List[List[Dict]]`: Nested list where each inner list contains detections for one frame

**Detection Dictionary Structure:**
```python
{
    'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
    'confidence': float,        # Detection confidence score
    'center': [center_x, center_y],  # Object center coordinates
    'class': int,               # Class ID (0-3)
    'class_name': str           # Human-readable class name
}
```

**Supported Classes:**
- `0`: 'ball'
- `1`: 'goalkeeper' 
- `2`: 'player'
- `3`: 'referee'

**Configuration:**
- Confidence threshold: 0.1 (very low for maximum recall)
- Processes all detected classes simultaneously

**Example:**
```python
detections = pipeline.detect_all_objects(frames, "broadcast")
print(f"Total detections: {sum(len(d) for d in detections)}")
```

---

### Feature Extraction

#### extract_visual_features

```python
def extract_visual_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray
```

Extract visual features from object bounding box region.

**Parameters:**
- `frame` (np.ndarray): Input video frame
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2]

**Returns:**
- `np.ndarray`: 26-dimensional visual feature vector

**Feature Composition:**
- **Color Histograms (24 dims)**: 8 bins each for R, G, B channels
- **Spatial Features (2 dims)**: Aspect ratio and normalized area

**Processing:**
- Crops object region from frame
- Computes normalized color histograms for illumination invariance
- Handles edge cases (small/empty crops)

**Example:**
```python
bbox = [100, 150, 200, 300]  # x1, y1, x2, y2
features = pipeline.extract_visual_features(frame, bbox)
print(f"Feature vector shape: {features.shape}")  # (26,)
```

#### extract_features

```python
def extract_features(self, detections: List[List[Dict]], frames: List[np.ndarray]) -> List[List[np.ndarray]]
```

Extract comprehensive features for all detections across all frames.

**Parameters:**
- `detections` (List[List[Dict]]): Detection results from `detect_all_objects`
- `frames` (List[np.ndarray]): Corresponding video frames

**Returns:**
- `List[List[np.ndarray]]`: Nested list of feature vectors per frame

**Feature Vector Composition (32 dimensions total):**
- **Visual features (26 dims)**: From `extract_visual_features`
- **Spatial features (6 dims)**:
  - Normalized center X coordinate
  - Normalized center Y coordinate  
  - Normalized width
  - Normalized height
  - Detection confidence score
  - Class ID

**Example:**
```python
features = pipeline.extract_features(detections, frames)
print(f"Features for frame 0: {len(features[0])} objects")
```

---

### Utility Methods

#### safe_extract_center

```python
def safe_extract_center(self, center: Union[List, Tuple, np.ndarray, float]) -> Tuple[float, float]
```

Safely extract center coordinates with robust error handling.

**Parameters:**
- `center` (Union[List, Tuple, np.ndarray, float]): Center coordinate data

**Returns:**
- `Tuple[float, float]`: (center_x, center_y) coordinates

**Error Handling:**
- Handles various input formats (list, tuple, numpy array, scalar)
- Provides fallback values (0.0, 0.0) for invalid inputs
- Flattens nested arrays automatically

**Example:**
```python
# Various input formats supported
center1 = pipeline.safe_extract_center([150, 200])
center2 = pipeline.safe_extract_center(np.array([320, 240]))
center3 = pipeline.safe_extract_center(320)  # Returns (320.0, 320.0)
```

---

## Matching Algorithms

### compute_similarity_matrix

```python
def compute_similarity_matrix(self, features1: List[np.ndarray], features2: List[np.ndarray]) -> np.ndarray
```

Compute similarity matrix between two sets of feature vectors.

**Parameters:**
- `features1` (List[np.ndarray]): Feature vectors from camera 1
- `features2` (List[np.ndarray]): Feature vectors from camera 2

**Returns:**
- `np.ndarray`: Similarity matrix of shape (len(features1), len(features2))

**Algorithm:**
1. **Base Similarity**: Cosine similarity between feature vectors
2. **Class-Aware Boosting**: 1.5x multiplier for same-class objects
3. **Normalization**: Ensures similarity values in reasonable range

**Mathematical Formula:**
```python
similarity[i,j] = cosine_similarity(feat1[i], feat2[j])
if class1[i] == class2[j]:
    similarity[i,j] *= 1.5
```

**Example:**
```python
sim_matrix = pipeline.compute_similarity_matrix(broadcast_feat, tacticam_feat)
print(f"Similarity matrix shape: {sim_matrix.shape}")
```

### match_objects_hungarian

```python
def match_objects_hungarian(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]
```

Perform optimal object matching using Hungarian algorithm.

**Parameters:**
- `similarity_matrix` (np.ndarray): Similarity matrix from `compute_similarity_matrix`

**Returns:**
- `List[Tuple[int, int, float]]`: List of matches as (idx1, idx2, similarity_score)

**Algorithm:**
1. **Cost Conversion**: Converts similarity to cost matrix (negative values)
2. **Linear Assignment**: Uses scipy's `linear_sum_assignment` for optimal matching
3. **Threshold Filtering**: Removes matches below minimum similarity (0.02)

**Complexity:**
- Time: O(n³) where n = min(objects_camera1, objects_camera2)
- Space: O(n²)

**Example:**
```python
matches = pipeline.match_objects_hungarian(similarity_matrix)
for obj1_idx, obj2_idx, score in matches:
    print(f"Object {obj1_idx} -> Object {obj2_idx} (similarity: {score:.3f})")
```

---

## Main Pipeline

### process_videos

```python
def process_videos(self, broadcast_path: str, tacticam_path: str) -> List[Dict]
```

Main processing function that orchestrates the entire pipeline.

**Parameters:**
- `broadcast_path` (str): Path to broadcast camera video
- `tacticam_path` (str): Path to tactical camera video

**Returns:**
- `List[Dict]`: List of frame results with match information

**Result Dictionary Structure:**
```python
{
    'frame': int,                  # Frame index
    'matches': List[Tuple],        # List of (obj1_idx, obj2_idx, similarity)
    'broadcast_count': int,        # Number of objects in broadcast frame
    'tacticam_count': int          # Number of objects in tacticam frame
}
```

**Pipeline Steps:**
1. **Frame Extraction**: Sample frames from both videos
2. **Object Detection**: Detect all objects in each frame
3. **Feature Extraction**: Compute comprehensive features
4. **Cross-Camera Matching**: Match objects between camera views
5. **Result Compilation**: Aggregate results with statistics

**Example:**
```python
results = pipeline.process_videos(
    '/path/to/broadcast.mp4',
    '/path/to/tacticam.mp4'
)

print(f"Processed {len(results)} frames")
for result in results[:5]:  # Show first 5 results
    print(f"Frame {result['frame']}: {len(result['matches'])} matches")
```

---

## Output Generation

### create_comprehensive_output_video

```python
def create_comprehensive_output_video(
    pipeline: PlayerReIDPipeline,
    results: List[Dict],
    broadcast_video_path: str,
    tacticam_video_path: str,
    output_path: str = 'all_classes_output.mp4'
) -> bool
```

Generate comprehensive output video with visualizations.

**Parameters:**
- `pipeline` (PlayerReIDPipeline): Initialized pipeline instance
- `results` (List[Dict]): Results from `process_videos`
- `broadcast_video_path` (str): Path to broadcast video
- `tacticam_video_path` (str): Path to tactical video
- `output_path` (str, optional): Output video path

**Returns:**
- `bool`: True if video creation successful, False otherwise

**Video Features:**
- **Side-by-side Layout**: Broadcast and tactical cameras
- **Color-coded Classes**:
  - Players: Green (0, 255, 0)
  - Goalkeepers: Red (255, 0, 0)
  - Referees: Blue (0, 0, 255)
  - Ball: Yellow (255, 255, 0)
- **Match Visualization**: Bright colors for matched objects
- **Information Overlay**: Frame statistics and object counts
- **Class Legend**: Bottom overlay showing class colors

**Technical Specifications:**
- **Output Format**: MP4 (H.264)
- **Frame Rate**: 3 FPS (for detailed viewing)
- **Resolution**: 2x original width (side-by-side)
- **Codec**: 'mp4v'

**Example:**
```python
success = create_comprehensive_output_video(
    pipeline, results,
    '/path/to/broadcast.mp4',
    '/path/to/tacticam.mp4',
    '/path/to/output.mp4'
)

if success:
    print("✅ Video created successfully")
else:
    print("❌ Video creation failed")
```

---

## Configuration Constants

### Detection Configuration

```python
# Model Classes
MODEL_CLASSES = {
    0: 'ball',
    1: 'goalkeeper',
    2: 'player',
    3: 'referee'
}

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.1  # Very low for maximum recall
NMS_THRESHOLD = 0.45        # Non-maximum suppression
MAX_DETECTIONS = 100        # Per frame limit
```

### Feature Configuration

```python
# Feature Extraction
COLOR_HISTOGRAM_BINS = 8    # Bins per color channel
SPATIAL_FEATURE_DIMS = 6    # Spatial feature dimensions
VISUAL_FEATURE_DIMS = 26    # Visual feature dimensions
TOTAL_FEATURE_DIMS = 32     # Total feature vector size
```

### Matching Configuration

```python
# Matching Parameters
SIMILARITY_THRESHOLD = 0.02    # Minimum similarity for valid match
CLASS_BOOST_FACTOR = 1.5       # Multiplier for same-class matches
FRAME_SAMPLING_RATIO = 0.5     # Middle 50% of video content
MAX_FRAMES_PER_VIDEO = 30      # Maximum frames to process
```

### Video Output Configuration

```python
# Video Generation
OUTPUT_FPS = 3                   # Slow FPS for detailed viewing
VIDEO_CODEC = 'mp4v'             # Output codec
MATCH_HIGHLIGHT_THICKNESS = 4    # Bounding box thickness for matches
DETECTION_THICKNESS = 2          # Regular detection thickness

# Color Scheme
CLASS_COLORS = {
    'player': (0, 255, 0),       # Green
    'goalkeeper': (255, 0, 0),   # Red
    'referee': (0, 0, 255),      # Blue
    'ball': (255, 255, 0),       # Yellow
    'person': (255, 0, 255),     # Magenta
}

MATCH_COLORS = [
    (0, 255, 255),      # Cyan
    (255, 165, 0),      # Orange
    (128, 0, 128),      # Purple
    (0, 128, 0),        # Dark Green
    (128, 128, 0),      # Olive
    (255, 192, 203),    # Pink
    (165, 42, 42),      # Brown
    (0, 255, 127)       # Spring Green
]
```

---

## Error Handling

### Common Exceptions

#### Video Processing Errors

```python
# File not found
FileNotFoundError: Video file doesn't exist

# Video cannot be opened
RuntimeError: Cannot open video file

# No frames extracted
ValueError: No frames found in video
```

#### Model Loading Errors

```python
# Model file corrupted
RuntimeError: PytorchStreamReader failed reading zip archive

# Model file not found
FileNotFoundError: Model file doesn't exist

# Incompatible model format
TypeError: Model format not supported
```

#### Processing Errors

```python
# Empty detection results
ValueError: No detections found

# Feature extraction failure
RuntimeError: Feature extraction failed

# Matching algorithm failure
LinAlgError: Singular matrix in Hungarian algorithm
```

### Error Handling Best Practices

```python
try:
    pipeline = PlayerReIDPipeline('/path/to/model.pt')
    results = pipeline.process_videos('video1.mp4', 'video2.mp4')
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Performance Considerations

### Memory Management

- **Frame Storage**: Frames are stored in memory for processing
- **Feature Vectors**: 32-dimensional vectors per detection
- **Similarity Matrix**: O(n×m) memory where n,m are detection counts

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Frame Extraction | O(n) | O(n × H × W × C) |
| Object Detection | O(n × m) | O(n × m) |
| Feature Extraction | O(d × f) | O(d × f) |
| Similarity Matrix | O(p × q × f) | O(p × q) |
| Hungarian Matching | O(min(p,q)³) | O(p × q) |

Where:
- n = number of frames
- m = average detections per frame  
- d = total detections
- f = feature dimensions
- p, q = detections per camera
- H, W, C = frame height, width, channels

### Optimization Tips

1. **Reduce Frame Count**: Lower `max_frames` parameter for faster processing
2. **GPU Acceleration**: Ensure CUDA is available for YOLO inference
3. **Batch Processing**: Process multiple frames simultaneously when possible
4. **Memory Cleanup**: Use `torch.cuda.empty_cache()` after processing

---

## Integration Examples

### Basic Usage

```python
# Initialize pipeline
pipeline = PlayerReIDPipeline('/path/to/best.pt')

# Process videos
results = pipeline.process_videos(
    broadcast_path='/path/to/broadcast.mp4',
    tacticam_path='/path/to/tacticam.mp4'
)

# Generate output video
success = create_comprehensive_output_video(
    pipeline, results,
    '/path/to/broadcast.mp4',
    '/path/to/tacticam.mp4',
    '/path/to/output.mp4'
)
```

### Advanced Configuration

```python
# Custom frame extraction
pipeline = PlayerReIDPipeline('/path/to/model.pt')
broadcast_frames = pipeline.extract_frames('/path/to/broadcast.mp4', max_frames=50)
tacticam_frames = pipeline.extract_frames('/path/to/tacticam.mp4', max_frames=50)

# Manual processing steps
broadcast_detections = pipeline.detect_all_objects(broadcast_frames, "broadcast")
tacticam_detections = pipeline.detect_all_objects(tacticam_frames, "tacticam")

broadcast_features = pipeline.extract_features(broadcast_detections, broadcast_frames)
tacticam_features = pipeline.extract_features(tacticam_detections, tacticam_frames)

# Frame-by-frame matching
for i in range(min(len(broadcast_features), len(tacticam_features))):
    sim_matrix = pipeline.compute_similarity_matrix(
        broadcast_features[i], tacticam_features[i]
    )
    matches = pipeline.match_objects_hungarian(sim_matrix)
    print(f"Frame {i}: {len(matches)} matches")
```

### Batch Processing

```python
def process_multiple_videos(video_pairs, model_path):
    """Process multiple video pairs efficiently"""
    pipeline = PlayerReIDPipeline(model_path)
    
    all_results = {}
    for pair_id, (broadcast_path, tacticam_path) in enumerate(video_pairs):
        print(f"Processing pair {pair_id + 1}/{len(video_pairs)}")
        
        results = pipeline.process_videos(broadcast_path, tacticam_path)
        all_results[pair_id] = results
        
        # Create output video
        output_path = f'/path/to/output_pair_{pair_id}.mp4'
        create_comprehensive_output_video(
            pipeline, results, broadcast_path, tacticam_path, output_path
        )

    return all_results
```