import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_annotations(csv_file_path):
    """
    Comprehensive analysis of video annotation data
    """
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    print("="*80)
    print("📊 VIDEO ANNOTATION ANALYSIS REPORT")
    print("="*80)
    
    # Basic dataset info
    print(f"\n📋 DATASET OVERVIEW")
    print(f"Total annotations: {len(df)}")
    print(f"Unique videos: {df['artifact_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Content type analysis
    print(f"\n🎬 CONTENT TYPE DISTRIBUTION")
    content_counts = df['content_type'].value_counts()
    for content_type, count in content_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{content_type}: {count} ({percentage:.1f}%)")
    
    # Page type analysis
    print(f"\n📱 PAGE TYPE DISTRIBUTION")
    page_counts = df['page_type'].value_counts()
    for page_type, count in page_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{page_type}: {count} ({percentage:.1f}%)")
    
    return df

def analyze_classification_agreement(df):
    """
    Analyze agreement between original AI classification and human corrections
    """
    print(f"\n🎯 CLASSIFICATION AGREEMENT ANALYSIS")
    print("-" * 50)
    
    # Calculate agreement
    df['agreement'] = df['original_classification'] == df['corrected_classification']
    
    total_annotations = len(df)
    agreements = df['agreement'].sum()
    disagreements = total_annotations - agreements
    
    agreement_rate = (agreements / total_annotations) * 100
    
    print(f"Total annotations: {total_annotations}")
    print(f"Agreements: {agreements} ({agreement_rate:.1f}%)")
    print(f"Disagreements: {disagreements} ({100-agreement_rate:.1f}%)")
    
    # Agreement by original classification
    print(f"\n📊 AGREEMENT BY ORIGINAL CLASSIFICATION")
    agreement_by_orig = df.groupby('original_classification')['agreement'].agg(['count', 'sum', 'mean']).round(3)
    agreement_by_orig.columns = ['Total', 'Agreements', 'Agreement_Rate']
    agreement_by_orig['Agreement_Rate'] = agreement_by_orig['Agreement_Rate'] * 100
    print(agreement_by_orig)
    
    # Classification changes analysis
    print(f"\n🔄 CLASSIFICATION CHANGES")
    changes = df[df['agreement'] == False]
    if len(changes) > 0:
        change_summary = changes.groupby(['original_classification', 'corrected_classification']).size()
        print("Original → Corrected (Count):")
        for (orig, corr), count in change_summary.items():
            print(f"  {orig} → {corr}: {count}")
    
    # Detailed correction analysis
    print(f"\n📝 CORRECTION DETAILS")
    correction_counts = df['classification_check'].value_counts()
    for correction, count in correction_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{correction}: {count} ({percentage:.1f}%)")
    
    return agreement_rate

def analyze_frame_detection(df):
    """
    Analyze frame detection quality (using Frame Caption Quality as indicator)
    """
    print(f"\n🎞️ FRAME DETECTION ANALYSIS")
    print("-" * 50)
    
    # Frame detection quality distribution
    frame_quality_counts = df['image_caption_quality'].value_counts()
    print(f"Frame Detection Quality Distribution:")
    for quality, count in frame_quality_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    # Frame detection by content type
    print(f"\n📱 FRAME DETECTION BY CONTENT TYPE")
    frame_by_content = pd.crosstab(df['content_type'], df['image_caption_quality'], margins=True)
    print(frame_by_content)
    
    # Frame detection by page type
    print(f"\n🎯 FRAME DETECTION BY PAGE TYPE")
    frame_by_page = pd.crosstab(df['page_type'], df['image_caption_quality'], margins=True)
    print(frame_by_page)
    
    # Frame number analysis
    if 'frame_number' in df.columns:
        print(f"\n🔢 FRAME NUMBER ANALYSIS")
        frame_num_counts = df['frame_number'].value_counts().sort_index()
        print("Distribution of which frames were analyzed:")
        for frame_num, count in frame_num_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  Frame {frame_num}: {count} ({percentage:.1f}%)")

def analyze_data_quality(df):
    """
    Analyze data collection quality
    """
    print(f"\n📊 DATA COLLECTION QUALITY ANALYSIS")
    print("-" * 50)
    
    # Data quality distribution
    data_quality_counts = df['data_collection_quality'].value_counts()
    print(f"Data Collection Quality Distribution:")
    for quality, count in data_quality_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {quality}: {count} ({percentage:.1f}%)")
    
    # Data quality by content type
    print(f"\n📱 DATA QUALITY BY CONTENT TYPE")
    quality_by_content = pd.crosstab(df['content_type'], df['data_collection_quality'], margins=True)
    print(quality_by_content)

def analyze_caption_quality(df):
    """
    Analyze caption quality metrics
    """
    print(f"\n📝 CAPTION QUALITY ANALYSIS")
    print("-" * 50)
    
    # Caption quality metrics
    caption_metrics = ['caption_accuracy', 'caption_completeness', 'caption_relevance']
    
    for metric in caption_metrics:
        if metric in df.columns:
            print(f"\n{metric.replace('_', ' ').title()} Distribution:")
            metric_counts = df[metric].value_counts()
            for value, count in metric_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
    
    # Caption quality scores
    if 'caption_quality_score' in df.columns:
        print(f"\n🎯 CAPTION QUALITY SCORES")
        scores = df['caption_quality_score'].dropna()
        if len(scores) > 0:
            print(f"Mean score: {scores.mean():.2f}")
            print(f"Median score: {scores.median():.2f}")
            print(f"Score range: {scores.min():.0f} - {scores.max():.0f}")
            print(f"Score distribution:")
            score_dist = scores.value_counts().sort_index()
            for score, count in score_dist.items():
                percentage = (count / len(scores)) * 100
                print(f"  Score {score}: {count} ({percentage:.1f}%)")
    
    # Caption issues analysis
    if 'caption_issues' in df.columns:
        print(f"\n⚠️ CAPTION ISSUES ANALYSIS")
        issues_data = df[df['caption_issues'] != 'None']['caption_issues'].dropna()
        if len(issues_data) > 0:
            all_issues = []
            for issues_str in issues_data:
                if issues_str and issues_str != 'None':
                    issues = [issue.strip() for issue in issues_str.split(',')]
                    all_issues.extend(issues)
            
            if all_issues:
                issue_counts = Counter(all_issues)
                print(f"Most common caption issues:")
                for issue, count in issue_counts.most_common():
                    percentage = (count / len(issues_data)) * 100
                    print(f"  {issue}: {count} ({percentage:.1f}%)")

def create_visualizations(df):
    """
    Create visualizations for the analysis
    """
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Video Annotation Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Agreement Rate
    ax1 = axes[0, 0]
    agreement_data = df.groupby('original_classification')['agreement'].mean() * 100
    agreement_data.plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4'])
    ax1.set_title('Agreement Rate by Original Classification')
    ax1.set_ylabel('Agreement Rate (%)')
    ax1.set_xlabel('Original Classification')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Frame Detection Quality
    ax2 = axes[0, 1]
    frame_quality = df['image_caption_quality'].value_counts()
    frame_quality.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
    ax2.set_title('Frame Detection Quality Distribution')
    ax2.set_ylabel('')
    
    # 3. Data Collection Quality by Content Type
    ax3 = axes[1, 0]
    quality_crosstab = pd.crosstab(df['content_type'], df['data_collection_quality'])
    quality_crosstab.plot(kind='bar', ax=ax3, stacked=True, color=['#ff6b6b', '#4ecdc4'])
    ax3.set_title('Data Quality by Content Type')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Content Type')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Quality')
    
    # 4. Caption Quality Scores (if available)
    ax4 = axes[1, 1]
    if 'caption_quality_score' in df.columns:
        scores = df['caption_quality_score'].dropna()
        if len(scores) > 0:
            scores.hist(bins=10, ax=ax4, color='#95a5a6', alpha=0.7)
            ax4.set_title('Caption Quality Score Distribution')
            ax4.set_xlabel('Quality Score')
            ax4.set_ylabel('Frequency')
        else:
            ax4.text(0.5, 0.5, 'No Caption Quality\nScores Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Caption Quality Scores')
    else:
        ax4.text(0.5, 0.5, 'No Caption Quality\nScores Available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Caption Quality Scores')
    
    plt.tight_layout()
    plt.savefig('annotation_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visualizations saved as 'annotation_analysis_dashboard.png'")

def generate_summary_report(df, agreement_rate):
    """
    Generate a summary report
    """
    print(f"\n📋 SUMMARY REPORT")
    print("=" * 80)
    
    # Key metrics
    total_annotations = len(df)
    unique_videos = df['artifact_id'].nunique()
    good_frame_detection = len(df[df['image_caption_quality'] == 'Good (1)'])
    good_data_quality = len(df[df['data_collection_quality'] == 'Good (1)'])
    
    print(f"📊 KEY METRICS:")
    print(f"  • Total annotations: {total_annotations}")
    print(f"  • Unique videos analyzed: {unique_videos}")
    print(f"  • Classification agreement rate: {agreement_rate:.1f}%")
    print(f"  • Good frame detection rate: {(good_frame_detection/total_annotations)*100:.1f}%")
    print(f"  • Good data quality rate: {(good_data_quality/total_annotations)*100:.1f}%")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if agreement_rate < 70:
        print(f"  • ⚠️ Low agreement rate ({agreement_rate:.1f}%) - Consider reviewing classification criteria")
    elif agreement_rate > 90:
        print(f"  • ✅ High agreement rate ({agreement_rate:.1f}%) - Classification system performing well")
    else:
        print(f"  • 👍 Moderate agreement rate ({agreement_rate:.1f}%) - System performance is acceptable")
    
    if (good_frame_detection/total_annotations) < 0.7:
        print(f"  • ⚠️ Frame detection needs improvement - Consider adjusting frame extraction logic")
    else:
        print(f"  • ✅ Frame detection performing well")
    
    if (good_data_quality/total_annotations) < 0.8:
        print(f"  • ⚠️ Data collection quality issues detected - Review data pipeline")
    else:
        print(f"  • ✅ Data collection quality is good")

def main():
    """
    Main function to run the complete analysis
    """
    print("🔍 Starting Video Annotation Analysis...")
    print("Please provide the path to your CSV file:")
    
    # You can modify this path or make it interactive
    csv_path = input("Enter CSV file path: ").strip()
    
    try:
        # Load and analyze data
        df = load_and_analyze_annotations(csv_path)
        
        # Prepare data for analysis
        df['agreement'] = df['original_classification'] == df['corrected_classification']
        
        # Run all analyses
        agreement_rate = analyze_classification_agreement(df)
        analyze_frame_detection(df)
        analyze_data_quality(df)
        analyze_caption_quality(df)
        
        # Create visualizations
        create_visualizations(df)
        
        # Generate summary report
        generate_summary_report(df, agreement_rate)
        
        print(f"\n✅ Analysis complete! Check the generated visualizations and summary above.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the CSV file at '{csv_path}'")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check your CSV file format and try again.")

if __name__ == "__main__":
    main()