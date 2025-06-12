"""
Cross-Camera Player Re-Identification Main Script
Entry point for running the complete pipeline with modular components
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.player_reid_pipeline import PlayerReIDPipeline, create_comprehensive_output_video, safe_model_loading
from src.visualization_suite import MetricsVisualizer, create_interactive_dashboard
from src.utils.feature_extraction import FeatureExtractor
from src.utils.matching_algorithms import MatchingAlgorithms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('player_reid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrossCameraPlayerMapping:
    """
    Main orchestrator class for cross-camera player mapping system
    Coordinates pipeline execution, visualization, and output generation
    """
    
    def __init__(self, 
                 model_path: str = "models/best.pt",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cross-camera player mapping system
        
        Args:
            model_path (str): Path to YOLOv8 model file
            config (dict): Configuration parameters
        """
        self.model_path = model_path
        self.config = config or self._get_default_config()
        
        # Initialize components
        logger.info("🚀 Initializing Cross-Camera Player Mapping System...")
        
        try:
            self.pipeline = PlayerReIDPipeline(model_path)
            self.visualizer = MetricsVisualizer(figsize=(14, 10))
            self.feature_extractor = FeatureExtractor(
                color_bins=self.config['feature_extraction']['color_bins'],
                normalize_features=self.config['feature_extraction']['normalize_features']
            )
            self.matcher = MatchingAlgorithms(
                similarity_threshold=self.config['matching']['similarity_threshold'],
                class_boost_factor=self.config['matching']['class_boost_factor']
            )
            
            logger.info("✅ System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters"""
        return {
            'video_processing': {
                'max_frames': 30,
                'frame_sampling_ratio': 0.5,
                'output_fps': 3
            },
            'detection': {
                'confidence_threshold': 0.1,
                'nms_threshold': 0.45
            },
            'feature_extraction': {
                'color_bins': 8,
                'normalize_features': True,
                'feature_weights': {
                    'color': 1.0,
                    'spatial': 1.0,
                    'geometric': 0.8
                }
            },
            'matching': {
                'similarity_threshold': 0.02,
                'class_boost_factor': 1.5,
                'matching_algorithm': 'hungarian'
            },
            'output': {
                'create_video': True,
                'create_analytics': True,
                'create_gif': True,
                'video_quality': 'high'
            }
        }
    
    def validate_inputs(self, broadcast_path: str, tacticam_path: str) -> bool:
        """
        Validate input video files and model
        
        Args:
            broadcast_path (str): Path to broadcast video
            tacticam_path (str): Path to tactical video
            
        Returns:
            bool: True if all inputs are valid
        """
        logger.info("🔍 Validating input files...")
        
        # Check video files
        for video_path, name in [(broadcast_path, "broadcast"), (tacticam_path, "tacticam")]:
            if not os.path.exists(video_path):
                logger.error(f"❌ {name.title()} video not found: {video_path}")
                return False
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                logger.error(f"❌ {name.title()} video file is empty: {video_path}")
                return False
            
            logger.info(f"✅ {name.title()} video validated: {file_size:,} bytes")
        
        # Check model file
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️  Model file not found: {self.model_path}")
            logger.info("Will attempt to use fallback model")
        else:
            model_size = os.path.getsize(self.model_path)
            logger.info(f"✅ Model file validated: {model_size:,} bytes")
        
        return True
    
    def process_videos(self, 
                      broadcast_path: str, 
                      tacticam_path: str,
                      output_dir: str = "videos/outputs") -> Dict[str, Any]:
        """
        Process videos through the complete pipeline
        
        Args:
            broadcast_path (str): Path to broadcast video
            tacticam_path (str): Path to tactical video
            output_dir (str): Output directory for results
            
        Returns:
            Dict with processing results and statistics
        """
        logger.info("🎬 Starting video processing pipeline...")
        
        # Validate inputs
        if not self.validate_inputs(broadcast_path, tacticam_path):
            raise ValueError("Input validation failed")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Process videos through main pipeline
            results = self.pipeline.process_videos(broadcast_path, tacticam_path)
            
            if not results:
                logger.warning("⚠️  No matches found in processing")
                return {'success': False, 'error': 'No matches found'}
            
            # Generate comprehensive statistics
            stats = self._generate_statistics(results)
            logger.info(f"📊 Processing complete: {stats['total_matches']} matches across {stats['total_frames']} frames")
            
            # Create outputs based on configuration
            output_files = {}
            
            if self.config['output']['create_video']:
                video_path = self._create_output_video(
                    results, broadcast_path, tacticam_path, output_dir
                )
                output_files['video'] = video_path
            
            if self.config['output']['create_analytics']:
                analytics_path = self._create_analytics_dashboard(results, output_dir)
                output_files['analytics'] = analytics_path
            
            return {
                'success': True,
                'results': results,
                'statistics': stats,
                'output_files': output_files,
                'pipeline_info': self.pipeline.get_pipeline_info()
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_output_video(self, 
                           results: List[Dict], 
                           broadcast_path: str,
                           tacticam_path: str, 
                           output_dir: str) -> str:
        """Create comprehensive output video"""
        logger.info("🎥 Creating output video...")
        
        timestamp = Path().ctime().replace(' ', '_').replace(':', '-')
        video_path = os.path.join(output_dir, f"cross_camera_output_{timestamp}.mp4")
        
        success = create_comprehensive_output_video(
            self.pipeline, results, broadcast_path, tacticam_path, video_path
        )
        
        if success:
            logger.info(f"✅ Output video created: {video_path}")
        else:
            logger.error("❌ Output video creation failed")
            
        return video_path if success else None
    
    def _create_analytics_dashboard(self, results: List[Dict], output_dir: str) -> str:
        """Create analytics dashboard"""
        logger.info("📊 Creating analytics dashboard...")
        
        try:
            # Extract data for visualization
            frame_data = []
            for result in results:
                frame_data.append({
                    'frame': result['frame'],
                    'matches': len(result['matches']),
                    'broadcast_objects': result['broadcast_count'],
                    'tacticam_objects': result['tacticam_count'],
                    'avg_similarity': sum(m[2] for m in result['matches']) / len(result['matches']) if result['matches'] else 0
                })
            
            # Create interactive dashboard
            dashboard_path = os.path.join(output_dir, "analytics_dashboard.html")
            create_interactive_dashboard(frame_data, dashboard_path)
            
            logger.info(f"✅ Analytics dashboard created: {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            logger.error(f"❌ Analytics dashboard creation failed: {e}")
            return None
    
    def _generate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive processing statistics"""
        total_matches = sum(len(result['matches']) for result in results)
        total_frames = len(results)
        
        similarities = []
        for result in results:
            similarities.extend([match[2] for match in result['matches']])
        
        stats = {
            'total_frames': total_frames,
            'total_matches': total_matches,
            'avg_matches_per_frame': total_matches / total_frames if total_frames > 0 else 0,
            'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'total_broadcast_detections': self.pipeline.get_pipeline_info().get('total_broadcast_detections', 0),
            'total_tacticam_detections': self.pipeline.get_pipeline_info().get('total_tacticam_detections', 0)
        }
        
        return stats
    
    def quick_demo(self, 
                   broadcast_path: str = "videos/broadcast.mp4",
                   tacticam_path: str = "videos/tacticam.mp4") -> Dict[str, Any]:
        """
        Quick demo function for testing and demonstrations
        
        Args:
            broadcast_path (str): Path to broadcast video
            tacticam_path (str): Path to tactical video
            
        Returns:
            Dict with demo results
        """
        logger.info("🎯 Running quick demo...")
        
        # Use reduced configuration for demo
        demo_config = self.config.copy()
        demo_config['video_processing']['max_frames'] = 15  # Faster processing
        demo_config['output']['create_gif'] = True
        
        # Temporarily update configuration
        original_max_frames = self.pipeline.max_frames
        self.pipeline.max_frames = demo_config['video_processing']['max_frames']
        
        try:
            results = self.process_videos(broadcast_path, tacticam_path)
            
            if results['success']:
                logger.info("🎉 Demo completed successfully!")
                logger.info(f"📊 Demo results: {results['statistics']}")
            else:
                logger.error(f"❌ Demo failed: {results.get('error', 'Unknown error')}")
            
            return results
            
        finally:
            # Restore original configuration
            self.pipeline.max_frames = original_max_frames

def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Cross-Camera Player Re-Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python player_reid.py --demo
  python player_reid.py -b videos/broadcast.mp4 -t videos/tacticam.mp4
  python player_reid.py -b broadcast.mp4 -t tacticam.mp4 -o results/ -m models/custom.pt
        """
    )
    
    parser.add_argument(
        '-b', '--broadcast',
        type=str,
        default="videos/broadcast.mp4",
        help="Path to broadcast camera video"
    )
    
    parser.add_argument(
        '-t', '--tacticam',
        type=str,
        default="videos/tacticam.mp4",
        help="Path to tactical camera video"
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default="models/best.pt",
        help="Path to YOLO model file"
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="videos/outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help="Run quick demo with sample data"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration if provided
        config = None
        if args.config and os.path.exists(args.config):
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        # Initialize system
        system = CrossCameraPlayerMapping(model_path=args.model, config=config)
        
        # Run processing
        if args.demo:
            results = system.quick_demo(args.broadcast, args.tacticam)
        else:
            results = system.process_videos(args.broadcast, args.tacticam, args.output)
        
        # Print results summary
        if results['success']:
            print("\n🎉 Processing completed successfully!")
            print(f"📊 Statistics: {results['statistics']}")
            if 'output_files' in results:
                print(f"📁 Output files: {results['output_files']}")
        else:
            print(f"\n❌ Processing failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

# Jupyter/Colab support functions
def run_colab_demo(broadcast_path: str = "/content/broadcast.mp4",
                   tacticam_path: str = "/content/tacticam.mp4",
                   model_path: str = "/content/best.pt"):
    """
    Convenient function for running in Google Colab
    
    Args:
        broadcast_path (str): Path to broadcast video
        tacticam_path (str): Path to tactical video  
        model_path (str): Path to model file
        
    Returns:
        Processing results
    """
    print("🎯 Running Cross-Camera Player Mapping Demo in Colab...")
    
    # Initialize system with Colab-optimized config
    colab_config = {
        'video_processing': {'max_frames': 20},
        'output': {
            'create_video': True,
            'create_analytics': True,
            'create_gif': True
        }
    }
    
    system = CrossCameraPlayerMapping(model_path=model_path, config=colab_config)
    results = system.process_videos(broadcast_path, tacticam_path, "/content/outputs")
    
    if results['success']:
        print("🎉 Demo completed! Check /content/outputs for results.")
        
        # Display results in Colab
        try:
            from IPython.display import Video, HTML, display
            if results['output_files'].get('video'):
                print("📹 Displaying output video:")
                display(Video(results['output_files']['video'], embed=True, width=800))
        except ImportError:
            print("💡 Install IPython to display videos inline")
    
    return results

if __name__ == "__main__":
    main()
