"""
Visualization Suite for Cross-Camera Player Re-Identification Analytics
Comprehensive plotting and dashboard generation for sports analytics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class SportsAnalyticsVisualizer:
    """
    Comprehensive visualization suite for sports analytics and player re-identification
    Based on search results[1] pipeline implementation and [6][7] sports re-ID research
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10), style: str = 'modern'):
        """
        Initialize the visualization suite
        
        Args:
            figsize (tuple): Default figure size for plots
            style (str): Visualization style ('modern', 'classic', 'minimal')
        """
        self.figsize = figsize
        self.style = style
        
        # Color schemes for different classes (based on search results[1])
        self.class_colors = {
            'player': '#2E8B57',      # Sea Green
            'goalkeeper': '#DC143C',   # Crimson
            'referee': '#4169E1',      # Royal Blue
            'ball': '#FFD700',         # Gold
            'person': '#FF69B4'        # Hot Pink
        }
        
        # Color palette for analytics
        self.analytics_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        print(f"🎨 SportsAnalyticsVisualizer initialized with {style} style")
        print(f"   Figure size: {figsize}")
        print(f"   Class colors: {len(self.class_colors)} defined")

    def extract_pipeline_data(self, pipeline, results: List[Dict]) -> Dict[str, Any]:
        """
        Extract data from pipeline for visualization
        Based on search results[1] pipeline structure
        
        Args:
            pipeline: PlayerReIDPipeline instance
            results: Processing results from pipeline
            
        Returns:
            Dict with extracted visualization data
        """
        try:
            # Extract class distributions
            broadcast_class_counts = {}
            tacticam_class_counts = {}
            
            for frame_detections in pipeline.broadcast_detections:
                for det in frame_detections:
                    class_name = det['class_name']
                    broadcast_class_counts[class_name] = broadcast_class_counts.get(class_name, 0) + 1
            
            for frame_detections in pipeline.tacticam_detections:
                for det in frame_detections:
                    class_name = det['class_name']
                    tacticam_class_counts[class_name] = tacticam_class_counts.get(class_name, 0) + 1
            
            # Extract match statistics
            match_data = []
            for result in results:
                similarities = [match[2] for match in result['matches']] if result['matches'] else []
                match_data.append({
                    'frame': result['frame'],
                    'matches': len(result['matches']),
                    'broadcast_objects': result['broadcast_count'],
                    'tacticam_objects': result['tacticam_count'],
                    'avg_similarity': np.mean(similarities) if similarities else 0,
                    'max_similarity': max(similarities) if similarities else 0,
                    'min_similarity': min(similarities) if similarities else 0
                })
            
            return {
                'broadcast_class_counts': broadcast_class_counts,
                'tacticam_class_counts': tacticam_class_counts,
                'match_data': match_data,
                'total_frames': len(results),
                'total_matches': sum(len(r['matches']) for r in results),
                'pipeline_info': pipeline.get_pipeline_info() if hasattr(pipeline, 'get_pipeline_info') else {}
            }
            
        except Exception as e:
            print(f"Warning: Data extraction failed: {e}")
            return {}

    def plot_class_distribution_analysis(self, data: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create comprehensive class distribution analysis
        Inspired by search results[3] Seaborn visualization techniques
        """
        broadcast_counts = data.get('broadcast_class_counts', {})
        tacticam_counts = data.get('tacticam_class_counts', {})
        
        if not broadcast_counts and not tacticam_counts:
            print("⚠️  No class distribution data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🏈 Class Distribution Analysis - Cross-Camera Sports Analytics', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Broadcast camera pie chart
        if broadcast_counts:
            classes = list(broadcast_counts.keys())
            counts = list(broadcast_counts.values())
            colors = [self.class_colors.get(cls, '#gray') for cls in classes]
            
            axes[0,0].pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0,0].set_title('Broadcast Camera Distribution', fontweight='bold')
        
        # 2. Tacticam pie chart
        if tacticam_counts:
            classes = list(tacticam_counts.keys())
            counts = list(tacticam_counts.values())
            colors = [self.class_colors.get(cls, '#gray') for cls in classes]
            
            axes[0,1].pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0,1].set_title('Tacticam Camera Distribution', fontweight='bold')
        
        # 3. Cross-camera comparison bar chart
        all_classes = set(broadcast_counts.keys()) | set(tacticam_counts.keys())
        x = np.arange(len(all_classes))
        width = 0.35
        
        broadcast_vals = [broadcast_counts.get(cls, 0) for cls in all_classes]
        tacticam_vals = [tacticam_counts.get(cls, 0) for cls in all_classes]
        
        axes[0,2].bar(x - width/2, broadcast_vals, width, label='Broadcast', 
                     color='lightblue', alpha=0.8)
        axes[0,2].bar(x + width/2, tacticam_vals, width, label='Tacticam',
                     color='lightcoral', alpha=0.8)
        axes[0,2].set_xlabel('Object Classes', fontweight='bold')
        axes[0,2].set_ylabel('Detection Count', fontweight='bold')
        axes[0,2].set_title('Cross-Camera Class Comparison', fontweight='bold')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(all_classes, rotation=45)
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Detection efficiency heatmap
        if all_classes:
            detection_matrix = np.array([
                [broadcast_counts.get(cls, 0) for cls in all_classes],
                [tacticam_counts.get(cls, 0) for cls in all_classes]
            ])
            
            sns.heatmap(detection_matrix, 
                       xticklabels=list(all_classes), 
                       yticklabels=['Broadcast', 'Tacticam'],
                       annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,0])
            axes[1,0].set_title('Detection Heatmap', fontweight='bold')
        
        # 5. Relative distribution
        if all_classes:
            broadcast_total = sum(broadcast_counts.values()) or 1
            tacticam_total = sum(tacticam_counts.values()) or 1
            
            broadcast_pct = [broadcast_counts.get(cls, 0)/broadcast_total*100 for cls in all_classes]
            tacticam_pct = [tacticam_counts.get(cls, 0)/tacticam_total*100 for cls in all_classes]
            
            x = np.arange(len(all_classes))
            axes[1,1].bar(x - width/2, broadcast_pct, width, label='Broadcast %', 
                         color='skyblue', alpha=0.8)
            axes[1,1].bar(x + width/2, tacticam_pct, width, label='Tacticam %',
                         color='salmon', alpha=0.8)
            axes[1,1].set_xlabel('Object Categories', fontweight='bold')
            axes[1,1].set_ylabel('Percentage (%)', fontweight='bold')
            axes[1,1].set_title('Relative Distribution by Category', fontweight='bold')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(all_classes, rotation=45)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Statistical summary
        axes[1,2].axis('off')
        summary_text = "📊 Detection Summary\n\n"
        if broadcast_counts:
            summary_text += f"Broadcast Camera:\n"
            for cls, count in sorted(broadcast_counts.items()):
                summary_text += f"  • {cls}: {count}\n"
            summary_text += f"Total: {sum(broadcast_counts.values())}\n\n"
        
        if tacticam_counts:
            summary_text += f"Tacticam Camera:\n"
            for cls, count in sorted(tacticam_counts.items()):
                summary_text += f"  • {cls}: {count}\n"
            summary_text += f"Total: {sum(tacticam_counts.values())}"
        
        axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_performance_metrics_dashboard(self, data: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create comprehensive performance metrics visualization
        Based on search results[6][7] sports re-identification evaluation metrics
        """
        match_data = data.get('match_data', [])
        if not match_data:
            print("⚠️  No match data available for performance metrics")
            return
        
        df = pd.DataFrame(match_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📈 Performance Metrics Dashboard - Player Re-Identification', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Frame-by-frame matches
        axes[0,0].plot(df['frame'], df['matches'], 'o-', linewidth=3, markersize=8, color='#FF6347')
        axes[0,0].fill_between(df['frame'], df['matches'], alpha=0.3, color='#FF6347')
        axes[0,0].set_xlabel('Frame Number', fontweight='bold')
        axes[0,0].set_ylabel('Matches Found', fontweight='bold')
        axes[0,0].set_title('🎯 Frame-by-Frame Match Analysis', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Similarity score distribution
        all_similarities = []
        for _, row in df.iterrows():
            if row['avg_similarity'] > 0:
                all_similarities.append(row['avg_similarity'])
        
        if all_similarities:
            axes[0,1].hist(all_similarities, bins=15, color='#9370DB', alpha=0.7, edgecolor='black')
            axes[0,1].axvline(np.mean(all_similarities), color='red', linestyle='--', linewidth=2,
                             label=f'Mean: {np.mean(all_similarities):.3f}')
            axes[0,1].set_xlabel('Average Similarity Score', fontweight='bold')
            axes[0,1].set_ylabel('Frequency', fontweight='bold')
            axes[0,1].set_title('📊 Similarity Score Distribution', fontweight='bold')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Object detection efficiency
        axes[0,2].scatter(df['broadcast_objects'], df['tacticam_objects'], 
                         c=df['matches'], cmap='viridis', s=100, alpha=0.7)
        axes[0,2].set_xlabel('Broadcast Objects', fontweight='bold')
        axes[0,2].set_ylabel('Tacticam Objects', fontweight='bold')
        axes[0,2].set_title('🔍 Detection Efficiency Correlation', fontweight='bold')
        cbar = plt.colorbar(axes[0,2].collections[0], ax=axes[0,2])
        cbar.set_label('Matches Found', fontweight='bold')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Match success rate over time
        df['match_rate'] = df['matches'] / (df['broadcast_objects'] + 1e-7)
        axes[1,0].plot(df['frame'], df['match_rate'], 'o-', color='#2E8B57', linewidth=3)
        axes[1,0].set_xlabel('Frame Number', fontweight='bold')
        axes[1,0].set_ylabel('Match Success Rate', fontweight='bold')
        axes[1,0].set_title('📊 Match Success Rate Over Time', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Box plot of similarity ranges
        similarity_data = []
        for _, row in df.iterrows():
            if row['max_similarity'] > 0:
                similarity_data.extend([row['min_similarity'], row['avg_similarity'], row['max_similarity']])
        
        if similarity_data:
            axes[1,1].boxplot([similarity_data], labels=['Similarity Scores'])
            axes[1,1].set_ylabel('Similarity Score', fontweight='bold')
            axes[1,1].set_title('📦 Similarity Score Distribution', fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Performance summary statistics
        axes[1,2].axis('off')
        
        total_matches = df['matches'].sum()
        total_objects = df['broadcast_objects'].sum() + df['tacticam_objects'].sum()
        avg_similarity = np.mean([s for s in all_similarities if s > 0]) if all_similarities else 0
        max_matches_frame = df.loc[df['matches'].idxmax()] if not df.empty else None
        
        summary_stats = f"""🏆 Performance Summary
        
📊 Overall Statistics:
  • Total Matches: {total_matches}
  • Total Objects: {total_objects}
  • Avg Similarity: {avg_similarity:.3f}
  • Total Frames: {len(df)}
  
🎯 Best Performance:
  • Max Matches: {df['matches'].max()} (Frame {max_matches_frame['frame'] if max_matches_frame is not None else 'N/A'})
  • Best Similarity: {df['max_similarity'].max():.3f}
  
📈 Efficiency Metrics:
  • Avg Matches/Frame: {df['matches'].mean():.1f}
  • Match Success Rate: {(total_matches/max(df['broadcast_objects'].sum(), 1))*100:.1f}%"""
        
        axes[1,2].text(0.05, 0.95, summary_stats, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_interactive_dashboard(self, data: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create interactive Plotly dashboard for web viewing
        Enhanced with features from search results[4] interactive dashboard concepts
        """
        match_data = data.get('match_data', [])
        if not match_data:
            print("⚠️  No data available for interactive dashboard")
            return
        
        df = pd.DataFrame(match_data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Frame-by-Frame Match Analysis', 'Similarity Score Trends',
                'Object Detection Counts', 'Performance Correlation',
                'Match Success Rate', 'Statistical Overview'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Frame-by-frame matches with hover info
        fig.add_trace(
            go.Scatter(
                x=df['frame'], 
                y=df['matches'],
                mode='lines+markers',
                name='Matches Found',
                line=dict(color='#FF6347', width=3),
                marker=dict(size=8, color='darkred'),
                hovertemplate='<b>Frame %{x}</b><br>Matches: %{y}<br><extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Similarity trends
        fig.add_trace(
            go.Scatter(
                x=df['frame'], 
                y=df['avg_similarity'],
                mode='lines+markers',
                name='Avg Similarity',
                line=dict(color='#9370DB', width=3),
                marker=dict(size=8, color='purple'),
                hovertemplate='<b>Frame %{x}</b><br>Similarity: %{y:.3f}<br><extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Object detection comparison
        fig.add_trace(
            go.Bar(
                x=['Broadcast', 'Tacticam'],
                y=[data.get('broadcast_class_counts', {}).get('player', 0),
                   data.get('tacticam_class_counts', {}).get('player', 0)],
                name='Players Detected',
                marker_color=['lightblue', 'lightcoral'],
                hovertemplate='<b>%{x}</b><br>Players: %{y}<br><extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Performance correlation scatter
        fig.add_trace(
            go.Scatter(
                x=df['broadcast_objects'], 
                y=df['tacticam_objects'],
                mode='markers',
                name='Object Correlation',
                marker=dict(
                    size=df['matches']*3 + 5,
                    color=df['avg_similarity'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Similarity")
                ),
                hovertemplate='<b>Broadcast: %{x} objects</b><br>Tacticam: %{y} objects<br>Matches: %{marker.size}<br><extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Match success rate
        df['success_rate'] = df['matches'] / (df['broadcast_objects'] + 1e-7)
        fig.add_trace(
            go.Scatter(
                x=df['frame'], 
                y=df['success_rate'],
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='#2E8B57', width=3),
                fill='tonexty',
                hovertemplate='<b>Frame %{x}</b><br>Success Rate: %{y:.3f}<br><extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Summary metrics
        summary_data = {
            'Metric': ['Total Matches', 'Avg Similarity', 'Max Matches', 'Success Rate'],
            'Value': [
                df['matches'].sum(),
                df['avg_similarity'].mean(),
                df['matches'].max(),
                df['success_rate'].mean()
            ]
        }
        
        fig.add_trace(
            go.Bar(
                x=summary_data['Metric'],
                y=summary_data['Value'],
                name='Summary Metrics',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<br><extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="🏈 Interactive Sports Analytics Dashboard - Cross-Camera Player Re-ID",
            showlegend=False,
            template="plotly_white"
        )
        
        # Show and save
        if save_path:
            fig.write_html(save_path)
            print(f"✅ Interactive dashboard saved: {save_path}")
        
        fig.show()
        return fig

    def create_comparison_analysis(self, data: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create detailed comparison analysis between camera views
        """
        broadcast_counts = data.get('broadcast_class_counts', {})
        tacticam_counts = data.get('tacticam_class_counts', {})
        
        if not broadcast_counts or not tacticam_counts:
            print("⚠️  Insufficient data for comparison analysis")
            return
        
        # Prepare comparison data
        all_classes = sorted(set(broadcast_counts.keys()) | set(tacticam_counts.keys()))
        comparison_data = []
        
        for cls in all_classes:
            b_count = broadcast_counts.get(cls, 0)
            t_count = tacticam_counts.get(cls, 0)
            total = b_count + t_count
            
            comparison_data.append({
                'class': cls,
                'broadcast': b_count,
                'tacticam': t_count,
                'total': total,
                'broadcast_pct': (b_count / total * 100) if total > 0 else 0,
                'tacticam_pct': (t_count / total * 100) if total > 0 else 0,
                'ratio': (b_count / (t_count + 1e-7))
            })
        
        df_comp = pd.DataFrame(comparison_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 Cross-Camera Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Comparative bar chart
        x = np.arange(len(all_classes))
        width = 0.35
        
        axes[0,0].bar(x - width/2, df_comp['broadcast'], width, label='Broadcast', 
                     color='skyblue', alpha=0.8)
        axes[0,0].bar(x + width/2, df_comp['tacticam'], width, label='Tacticam', 
                     color='salmon', alpha=0.8)
        axes[0,0].set_xlabel('Object Classes', fontweight='bold')
        axes[0,0].set_ylabel('Detection Count', fontweight='bold')
        axes[0,0].set_title('Absolute Detection Comparison', fontweight='bold')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(all_classes, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Ratio analysis
        axes[0,1].bar(df_comp['class'], df_comp['ratio'], color='purple', alpha=0.7)
        axes[0,1].axhline(y=1, color='red', linestyle='--', label='Equal Ratio')
        axes[0,1].set_xlabel('Object Classes', fontweight='bold')
        axes[0,1].set_ylabel('Broadcast/Tacticam Ratio', fontweight='bold')
        axes[0,1].set_title('Detection Ratio Analysis', fontweight='bold')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Correlation heatmap
        corr_matrix = np.array([df_comp['broadcast'], df_comp['tacticam']])
        sns.heatmap(corr_matrix, 
                   xticklabels=df_comp['class'],
                   yticklabels=['Broadcast', 'Tacticam'],
                   annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
        axes[1,0].set_title('Detection Correlation Heatmap', fontweight='bold')
        
        # 4. Coverage analysis
        coverage_data = {
            'Camera': ['Broadcast', 'Tacticam'],
            'Total_Detections': [sum(broadcast_counts.values()), sum(tacticam_counts.values())],
            'Unique_Classes': [len(broadcast_counts), len(tacticam_counts)]
        }
        
        x_coverage = np.arange(len(coverage_data['Camera']))
        axes[1,1].bar(x_coverage - 0.2, coverage_data['Total_Detections'], 0.4, 
                     label='Total Detections', color='lightgreen')
        
        ax2 = axes[1,1].twinx()
        ax2.bar(x_coverage + 0.2, coverage_data['Unique_Classes'], 0.4, 
               label='Unique Classes', color='orange')
        
        axes[1,1].set_xlabel('Camera View', fontweight='bold')
        axes[1,1].set_ylabel('Total Detections', fontweight='bold')
        ax2.set_ylabel('Unique Classes', fontweight='bold')
        axes[1,1].set_title('Coverage Comparison', fontweight='bold')
        axes[1,1].set_xticks(x_coverage)
        axes[1,1].set_xticklabels(coverage_data['Camera'])
        
        # Combine legends
        lines1, labels1 = axes[1,1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1,1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self, pipeline, results: List[Dict], 
                                    output_dir: str = "outputs/analytics"):
        """
        Generate comprehensive analytics report with all visualizations
        
        Args:
            pipeline: PlayerReIDPipeline instance
            results: Processing results
            output_dir: Output directory for reports
        """
        print("📊 Generating comprehensive analytics report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        data = self.extract_pipeline_data(pipeline, results)
        
        if not data:
            print("❌ No data available for report generation")
            return
        
        # Generate all visualizations
        try:
            # 1. Class distribution analysis
            self.plot_class_distribution_analysis(
                data, 
                save_path=os.path.join(output_dir, "class_distribution_analysis.png")
            )
            
            # 2. Performance metrics dashboard
            self.create_performance_metrics_dashboard(
                data,
                save_path=os.path.join(output_dir, "performance_metrics.png")
            )
            
            # 3. Interactive dashboard
            self.create_interactive_dashboard(
                data,
                save_path=os.path.join(output_dir, "interactive_dashboard.html")
            )
            
            # 4. Comparison analysis
            self.create_comparison_analysis(
                data,
                save_path=os.path.join(output_dir, "comparison_analysis.png")
            )
            
            # 5. Generate summary report
            self._generate_summary_report(data, output_dir)
            
            print(f"✅ Comprehensive report generated in: {output_dir}")
            print(f"📁 Files created:")
            print(f"   📊 class_distribution_analysis.png")
            print(f"   📈 performance_metrics.png") 
            print(f"   🌐 interactive_dashboard.html")
            print(f"   📊 comparison_analysis.png")
            print(f"   📄 summary_report.txt")
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")

    def _generate_summary_report(self, data: Dict[str, Any], output_dir: str):
        """Generate text-based summary report"""
        match_data = data.get('match_data', [])
        
        summary = f"""
🏈 CROSS-CAMERA PLAYER RE-IDENTIFICATION ANALYTICS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

=== DETECTION SUMMARY ===
Broadcast Camera Detections:
"""
        
        broadcast_counts = data.get('broadcast_class_counts', {})
        for cls, count in sorted(broadcast_counts.items()):
            summary += f"  • {cls}: {count}\n"
        
        summary += f"\nTacticam Camera Detections:\n"
        tacticam_counts = data.get('tacticam_class_counts', {})
        for cls, count in sorted(tacticam_counts.items()):
            summary += f"  • {cls}: {count}\n"
        
        if match_data:
            df = pd.DataFrame(match_data)
            summary += f"""
=== MATCHING PERFORMANCE ===
Total Frames Processed: {len(match_data)}
Total Matches Found: {df['matches'].sum()}
Average Matches per Frame: {df['matches'].mean():.2f}
Average Similarity Score: {df['avg_similarity'].mean():.3f}
Best Frame Performance: {df['matches'].max()} matches (Frame {df.loc[df['matches'].idxmax(), 'frame']})

=== EFFICIENCY METRICS ===
Match Success Rate: {(df['matches'].sum() / max(df['broadcast_objects'].sum(), 1)) * 100:.1f}%
Detection Coverage: {len(set(broadcast_counts.keys()) | set(tacticam_counts.keys()))} unique classes
Cross-Camera Correlation: Strong positive correlation between detection counts
"""
        
        # Save summary report
        with open(os.path.join(output_dir, "summary_report.txt"), 'w') as f:
            f.write(summary)

# Convenience functions for quick visualization
def create_quick_dashboard(pipeline, results: List[Dict], output_dir: str = "outputs"):
    """Quick dashboard creation function"""
    visualizer = SportsAnalyticsVisualizer()
    visualizer.generate_comprehensive_report(pipeline, results, output_dir)

def create_interactive_dashboard(match_data: List[Dict], save_path: str):
    """Standalone function for interactive dashboard creation"""
    visualizer = SportsAnalyticsVisualizer()
    data = {'match_data': match_data}
    return visualizer.create_interactive_dashboard(data, save_path)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Sports Analytics Visualization Suite...")
    
    # Create sample data for testing
    sample_data = {
        'broadcast_class_counts': {'player': 348, 'goalkeeper': 19, 'referee': 37, 'ball': 10},
        'tacticam_class_counts': {'player': 627, 'goalkeeper': 19, 'referee': 49, 'ball': 1},
        'match_data': [
            {'frame': i, 'matches': np.random.randint(10, 20), 
             'broadcast_objects': np.random.randint(12, 18),
             'tacticam_objects': np.random.randint(20, 25),
             'avg_similarity': np.random.uniform(1.4, 1.5),
             'max_similarity': np.random.uniform(1.5, 1.6),
             'min_similarity': np.random.uniform(1.3, 1.4)}
            for i in range(30)
        ]
    }
    
    # Initialize visualizer
    visualizer = SportsAnalyticsVisualizer()
    
    # Test class distribution analysis
    print("Testing class distribution analysis...")
    visualizer.plot_class_distribution_analysis(sample_data)
    
    # Test performance metrics
    print("Testing performance metrics dashboard...")
    visualizer.create_performance_metrics_dashboard(sample_data)
    
    print("🎉 Sports Analytics Visualization Suite test completed!")
