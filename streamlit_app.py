import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Student Assessment Analysis", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
    h1 {color: #1f77b4;}
    h2 {color: #2ca02c;}
    h3 {color: #ff7f0e;}
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def load_data(file):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert time columns
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], format='%d-%m-%Y %H:%M', errors='coerce')
    if 'submit_time' in df.columns:
        df['submit_time'] = pd.to_datetime(df['submit_time'], format='%d-%m-%Y %H:%M', errors='coerce')
    
    # Calculate additional metrics
    df['time_taken_minutes'] = df['total_time_taken'] / 60
    df['marks_per_minute'] = df['marks_obtained'] / df['time_taken_minutes']
    df['correct_incorrect_ratio'] = df['correct_count'] / (df['incorrect_count'] + 1)
    df['attempt_rate'] = (df['max_attempted_score'] / df['max_test_score']) * 100
    df['skip_rate'] = (df['marks_skipped'] / df['max_test_score']) * 100
    
    # Extract hour from start_time
    if 'start_time' in df.columns:
        df['start_hour'] = df['start_time'].dt.hour
    
    return df

def create_distribution_plot(data, column, title, bins=20):
    """Create a histogram with distribution"""
    fig = px.histogram(data, x=column, nbins=bins, title=title,
                      labels={column: column.replace('_', ' ').title()})
    fig.update_layout(showlegend=False)
    return fig

def create_bar_chart(data, x, y, title, orientation='v'):
    """Create a bar chart"""
    if orientation == 'h':
        fig = px.bar(data, x=y, y=x, orientation='h', title=title)
    else:
        fig = px.bar(data, x=x, y=y, title=title)
    return fig

# Main application
st.title("📊 Talentio Data Analysis Dashboard")
st.markdown("### Comprehensive Analysis of Student Performance Metrics")

# File upload
uploaded_file = st.file_uploader("Upload Student Assessment CSV File", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    st.success(f"✅ Data loaded successfully! Total students: {len(df)}")
    
    # Create tabs for different analysis categories
    tabs = st.tabs([
        "📈 Performance",
        "⏱️ Time Analysis", 
        "🏫 Demographics",
        "🎯 Behavior",
        "💻 Technical Section",
        "🔗 Correlations",
        "🏆 Benchmarking",
        "⚡ Efficiency",
        "📊 Statistics"
    ])
    
    # TAB 1: PERFORMANCE ANALYSIS
    with tabs[0]:
        st.header("Performance Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Score", f"{df['marks_obtained'].mean():.2f}")
        with col2:
            st.metric("Average Percentage", f"{df['percentage'].mean():.2f}%")
        with col3:
            st.metric("Pass Rate (>40%)", f"{(df['percentage'] > 40).sum() / len(df) * 100:.1f}%")
        with col4:
            st.metric("Perfect Scores", f"{(df['percentage'] == 100).sum()}")
        
        st.subheader("1. Overall Score Distribution")
        fig = create_distribution_plot(df, 'percentage', 'Distribution of Student Percentages')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("2. Pass/Fail Analysis")
        thresholds = [40, 50, 60, 70, 80]
        pass_rates = []
        for thresh in thresholds:
            pass_rate = (df['percentage'] >= thresh).sum() / len(df) * 100
            pass_rates.append(pass_rate)
        
        pass_df = pd.DataFrame({'Threshold': [f'{t}%' for t in thresholds], 'Pass Rate': pass_rates})
        fig = px.bar(pass_df, x='Threshold', y='Pass Rate', title='Pass Rates at Different Thresholds')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("3. Top 10 Performers")
            top_performers = df.nlargest(10, 'marks_obtained')[['Full Name', 'marks_obtained', 'percentage', 'College Name']]
            st.dataframe(top_performers, use_container_width=True)
        
        with col2:
            st.subheader("4. Low Performers (Bottom 10)")
            low_performers = df.nsmallest(10, 'marks_obtained')[['Full Name', 'marks_obtained', 'percentage', 'College Name']]
            st.dataframe(low_performers, use_container_width=True)
        
        st.subheader("5. Score Range Categorization")
        def categorize_score(pct):
            if pct >= 90: return 'Excellent'
            elif pct >= 75: return 'Good'
            elif pct >= 50: return 'Average'
            else: return 'Below Average'
        
        df['score_category'] = df['percentage'].apply(categorize_score)
        category_counts = df['score_category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index, 
                     title='Score Distribution by Category')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("6. Marks Gained vs Marks Lost Ratio")
        df['gain_loss_ratio'] = df['marks_gained'] / (df['marks_lost'] + 1)
        fig = px.histogram(df, x='gain_loss_ratio', nbins=30, 
                          title='Distribution of Marks Gained/Lost Ratio')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("7. Attempted vs Skipped Questions")
        attempt_skip_df = pd.DataFrame({
            'Category': ['Attempted', 'Skipped'],
            'Average': [df['max_attempted_score'].mean(), df['marks_skipped'].mean()]
        })
        fig = px.bar(attempt_skip_df, x='Category', y='Average', 
                     title='Average Attempted vs Skipped Marks')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("8. Avg Correct Answer %", f"{(df['correct_count'].sum() / (df['correct_count'].sum() + df['incorrect_count'].sum()) * 100):.2f}%")
        with col2:
            st.metric("9. Avg Incorrect Answer %", f"{(df['incorrect_count'].sum() / (df['correct_count'].sum() + df['incorrect_count'].sum()) * 100):.2f}%")
        
        st.subheader("10. Question Skip Rate Analysis")
        fig = create_distribution_plot(df, 'skip_rate', 'Distribution of Skip Rates (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("11. Correct to Incorrect Ratio")
        fig = px.box(df, y='correct_incorrect_ratio', 
                     title='Box Plot: Correct to Incorrect Answer Ratio')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("12. Attempt Completeness Patterns")
        fig = px.scatter(df, x='max_attempted_score', y='marks_obtained',
                        title='Attempted Score vs Marks Obtained',
                        trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("14. Perfect Scorers", f"{(df['percentage'] == 100).sum()}")
        with col2:
            st.metric("15. Zero Scores", f"{(df['marks_obtained'] == 0).sum()}")
        with col3:
            st.metric("19. Avg Marks", f"{df['marks_obtained'].mean():.2f}")
        
        st.subheader("20. Performance Variance Analysis")
        variance_metrics = pd.DataFrame({
            'Metric': ['Standard Deviation', 'Variance', 'Coefficient of Variation'],
            'Value': [
                df['percentage'].std(),
                df['percentage'].var(),
                (df['percentage'].std() / df['percentage'].mean()) * 100
            ]
        })
        st.dataframe(variance_metrics, use_container_width=True)
    
    # TAB 2: TIME-BASED ANALYSIS
    with tabs[1]:
        st.header("Time-Based Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Time (minutes)", f"{df['time_taken_minutes'].mean():.2f}")
        with col2:
            st.metric("Max Time", f"{df['time_taken_minutes'].max():.2f} min")
        with col3:
            st.metric("Min Time", f"{df['time_taken_minutes'].min():.2f} min")
        
        st.subheader("1. Time Distribution")
        fig = create_distribution_plot(df, 'time_taken_minutes', 
                                      'Distribution of Test Completion Time (minutes)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("2. Time vs Performance Correlation")
        fig = px.scatter(df, x='time_taken_minutes', y='percentage',
                        title='Time Taken vs Performance',
                        trendline='ols', labels={'time_taken_minutes': 'Time (minutes)'})
        st.plotly_chart(fig, use_container_width=True)
        
        correlation = df['time_taken_minutes'].corr(df['percentage'])
        st.info(f"Correlation coefficient: {correlation:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("3. Longest Time Takers")
            longest_time = df.nlargest(10, 'time_taken_minutes')[['Full Name', 'time_taken_minutes', 'percentage']]
            st.dataframe(longest_time, use_container_width=True)
        
        with col2:
            st.subheader("4. Shortest Time Takers")
            shortest_time = df.nsmallest(10, 'time_taken_minutes')[['Full Name', 'time_taken_minutes', 'percentage']]
            st.dataframe(shortest_time, use_container_width=True)
        
        st.subheader("5. Optimal Time Range Analysis")
        # Create time bins
        df['time_bin'] = pd.cut(df['time_taken_minutes'], bins=5)
        df['time_bin_str'] = df['time_bin'].astype(str)
        time_performance = df.groupby('time_bin_str')['percentage'].mean().reset_index()
        fig = px.bar(time_performance, x='time_bin_str', y='percentage',
                    title='Average Performance by Time Range',
                    labels={'time_bin_str': 'Time Range (minutes)'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("8. Time per Question")
        total_questions = df['correct_count'] + df['incorrect_count'] + df['na_count']
        df['time_per_question'] = df['time_taken_minutes'] / total_questions
        st.metric("Average Time per Question", f"{df['time_per_question'].mean():.2f} minutes")
        
        st.subheader("9. Time Efficiency (Marks per Minute)")
        fig = px.histogram(df, x='marks_per_minute', nbins=30,
                          title='Distribution of Marks Obtained per Minute')
        st.plotly_chart(fig, use_container_width=True)
        
        if 'start_hour' in df.columns:
            st.subheader("10. Test Submission Time Patterns")
            hourly_performance = df.groupby('start_hour')['percentage'].mean().reset_index()
            fig = px.line(hourly_performance, x='start_hour', y='percentage',
                         title='Average Performance by Start Hour',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("12. Peak Performance Hours")
            best_hour = hourly_performance.loc[hourly_performance['percentage'].idxmax()]
            st.success(f"Best performance at hour: {int(best_hour['start_hour'])}:00 with average {best_hour['percentage']:.2f}%")
    
    # TAB 3: DEMOGRAPHIC & INSTITUTIONAL ANALYSIS
    with tabs[2]:
        st.header("Demographic & Institutional Analysis")
        
        if 'College Name' in df.columns:
            st.subheader("1-4. College-wise Performance")
            college_stats = df.groupby('College Name').agg({
                'marks_obtained': 'mean',
                'percentage': ['mean', 'count'],
                'Full Name': 'count'
            }).round(2)
            college_stats.columns = ['Avg Marks', 'Avg Percentage', 'Count', 'Students']
            college_stats['Pass Rate %'] = df.groupby('College Name').apply(
                lambda x: (x['percentage'] > 40).sum() / len(x) * 100
            ).round(2)
            
            st.dataframe(college_stats.sort_values('Avg Percentage', ascending=False), 
                        use_container_width=True)
            
            fig = px.bar(college_stats.reset_index(), x='College Name', y='Avg Percentage',
                        title='Average Performance by College')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Department' in df.columns or 'College Department' in df.columns:
            dept_col = 'Department' if 'Department' in df.columns else 'College Department'
            
            st.subheader("5-8. Department-wise Performance")
            dept_stats = df.groupby(dept_col).agg({
                'marks_obtained': 'mean',
                'percentage': 'mean',
                'correct_count': 'mean',
                'incorrect_count': 'mean',
                'Full Name': 'count'
            }).round(2)
            dept_stats.columns = ['Avg Marks', 'Avg %', 'Avg Correct', 'Avg Incorrect', 'Students']
            dept_stats['Correct/Incorrect'] = (dept_stats['Avg Correct'] / 
                                               (dept_stats['Avg Incorrect'] + 1)).round(2)
            
            st.dataframe(dept_stats.sort_values('Avg %', ascending=False), 
                        use_container_width=True)
            
            fig = px.bar(dept_stats.reset_index(), x=dept_col, y='Avg %',
                        title='Average Percentage by Department')
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Section' in df.columns or 'College Section' in df.columns:
            sect_col = 'Section' if 'Section' in df.columns else 'College Section'
            
            st.subheader("9-11. Section-wise Performance")
            section_stats = df.groupby(sect_col).agg({
                'marks_obtained': 'mean',
                'percentage': 'mean',
                'Full Name': 'count'
            }).round(2)
            section_stats.columns = ['Avg Marks', 'Avg %', 'Students']
            
            st.dataframe(section_stats.sort_values('Avg %', ascending=False), 
                        use_container_width=True)
    
    # TAB 4: BEHAVIORAL & ENGAGEMENT ANALYSIS
    with tabs[3]:
        st.header("Behavioral & Engagement Analysis")
        
        if 'attempt_focus_change_count' in df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Focus Changes", f"{df['attempt_focus_change_count'].mean():.2f}")
            with col2:
                st.metric("Max Focus Changes", f"{df['attempt_focus_change_count'].max()}")
            with col3:
                st.metric("Zero Focus Changes", f"{(df['attempt_focus_change_count'] == 0).sum()}")
            
            st.subheader("1. Focus Change Distribution")
            fig = px.histogram(df, x='attempt_focus_change_count', nbins=20,
                             title='Distribution of Focus Changes')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("2. Focus Changes vs Performance")
            fig = px.scatter(df, x='attempt_focus_change_count', y='percentage',
                           title='Focus Changes vs Performance',
                           trendline='ols')
            st.plotly_chart(fig, use_container_width=True)
            
            correlation = df['attempt_focus_change_count'].corr(df['percentage'])
            st.info(f"Correlation: {correlation:.3f}")
            
            st.subheader("3. High Distraction Students (>10 focus changes)")
            high_distraction = df[df['attempt_focus_change_count'] > 10][
                ['Full Name', 'attempt_focus_change_count', 'percentage']
            ].sort_values('attempt_focus_change_count', ascending=False)
            st.dataframe(high_distraction, use_container_width=True)
            
            st.subheader("7. Zero Focus Change Performance")
            zero_focus = df[df['attempt_focus_change_count'] == 0]['percentage'].mean()
            other_focus = df[df['attempt_focus_change_count'] > 0]['percentage'].mean()
            
            comparison_df = pd.DataFrame({
                'Group': ['Zero Focus Changes', 'With Focus Changes'],
                'Avg Percentage': [zero_focus, other_focus]
            })
            fig = px.bar(comparison_df, x='Group', y='Avg Percentage',
                        title='Performance: Zero vs Non-Zero Focus Changes')
            st.plotly_chart(fig, use_container_width=True)
        
        if 'attempt_ending_method' in df.columns:
            st.subheader("8. Test Completion Method")
            ending_method_counts = df['attempt_ending_method'].value_counts()
            fig = px.pie(values=ending_method_counts.values, names=ending_method_counts.index,
                        title='Distribution of Test Ending Methods')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("10. Risk-Taking Behavior")
        df['risk_category'] = df['attempt_rate'].apply(
            lambda x: 'Aggressive (>90%)' if x > 90 else 
                     ('Moderate (70-90%)' if x > 70 else 'Conservative (<70%)')
        )
        risk_performance = df.groupby('risk_category')['percentage'].mean().reset_index()
        fig = px.bar(risk_performance, x='risk_category', y='percentage',
                    title='Performance by Risk-Taking Strategy')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("13. Strategic Answering Pattern")
        strategic = df[(df['skip_rate'] > 20) & (df['percentage'] > 70)]
        st.write(f"Students with strategic answering (>20% skip rate, >70% accuracy): {len(strategic)}")
        if len(strategic) > 0:
            st.dataframe(strategic[['Full Name', 'skip_rate', 'percentage', 'correct_count']], 
                        use_container_width=True)
    
    # TAB 5: TECHNICAL SECTION ANALYSIS
    with tabs[4]:
        st.header("Technical Section Specific Analysis")
        
        tech_cols = [col for col in df.columns if 'Technical MCQ' in col]
        
        if len(tech_cols) > 0:
            if 'Section - Technical MCQ - percentage' in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Technical MCQ %", 
                             f"{df['Section - Technical MCQ - percentage'].mean():.2f}%")
                with col2:
                    st.metric("Avg Overall %", f"{df['percentage'].mean():.2f}%")
                
                st.subheader("1. Technical MCQ Performance Distribution")
                fig = px.histogram(df, x='Section - Technical MCQ - percentage',
                                 nbins=20, title='Technical MCQ Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("2. Technical vs Overall Performance")
                fig = px.scatter(df, x='Section - Technical MCQ - percentage', 
                               y='percentage',
                               title='Technical MCQ % vs Overall %',
                               trendline='ols')
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Section - Technical MCQ - total_time_taken' in df.columns:
                st.subheader("3. Technical MCQ Time Analysis")
                df['tech_time_minutes'] = df['Section - Technical MCQ - total_time_taken'] / 60
                st.metric("Avg Time on Technical MCQ", 
                         f"{df['tech_time_minutes'].mean():.2f} minutes")
                
                fig = px.histogram(df, x='tech_time_minutes',
                                 title='Time Spent on Technical MCQ (minutes)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Technical MCQ specific columns found in the dataset")
    
    # TAB 6: CORRELATION ANALYSIS
    with tabs[5]:
        st.header("Correlation Analysis")
        
        st.subheader("Correlation Matrix")
        numeric_cols = ['correct_count', 'incorrect_count', 'marks_obtained', 
                       'marks_skipped', 'total_time_taken', 'percentage',
                       'attempt_focus_change_count', 'marks_gained', 'marks_lost']
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[available_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect='auto',
                       title='Correlation Heatmap',
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Key Correlations")
        
        correlations = []
        if 'correct_count' in df.columns and 'total_time_taken' in df.columns:
            corr = df['correct_count'].corr(df['total_time_taken'])
            correlations.append(('Correct Count vs Time', corr))
        
        if 'incorrect_count' in df.columns and 'attempt_focus_change_count' in df.columns:
            corr = df['incorrect_count'].corr(df['attempt_focus_change_count'])
            correlations.append(('Incorrect Count vs Focus Changes', corr))
        
        if 'marks_skipped' in df.columns and 'percentage' in df.columns:
            corr = df['marks_skipped'].corr(df['percentage'])
            correlations.append(('Marks Skipped vs Performance', corr))
        
        if correlations:
            corr_df = pd.DataFrame(correlations, columns=['Correlation Pair', 'Coefficient'])
            st.dataframe(corr_df, use_container_width=True)
    
    # TAB 7: BENCHMARKING
    with tabs[6]:
        st.header("Comparative & Benchmarking Analysis")
        
        st.subheader("1. Student Ranking")
        df['rank'] = df['percentage'].rank(ascending=False, method='min')
        top_20 = df.nsmallest(20, 'rank')[['rank', 'Full Name', 'percentage', 'marks_obtained']]
        st.dataframe(top_20, use_container_width=True)
        
        st.subheader("2. Percentile Analysis")
        df['percentile'] = df['percentage'].rank(pct=True) * 100
        
        fig = px.histogram(df, x='percentile', nbins=20,
                          title='Distribution of Student Percentiles')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("3. Above/Below Average Segmentation")
        avg_score = df['percentage'].mean()
        above_avg = (df['percentage'] >= avg_score).sum()
        below_avg = (df['percentage'] < avg_score).sum()
        
        seg_df = pd.DataFrame({
            'Segment': ['Above Average', 'Below Average'],
            'Count': [above_avg, below_avg]
        })
        fig = px.pie(seg_df, values='Count', names='Segment',
                    title=f'Performance Segmentation (Avg: {avg_score:.2f}%)')
        st.plotly_chart(fig, use_container_width=True)
        
        if 'College Name' in df.columns:
            st.subheader("5. Institutional Benchmarking")
            college_benchmark = df.groupby('College Name').agg({
                'percentage': ['mean', 'median', 'std'],
                'Full Name': 'count'
            }).round(2)
            college_benchmark.columns = ['Mean %', 'Median %', 'Std Dev', 'Students']
            st.dataframe(college_benchmark, use_container_width=True)
    
    # TAB 8: EFFICIENCY METRICS
    with tabs[7]:
        st.header("Efficiency & Effectiveness Metrics")
        
        st.subheader("1. Marks Gained per Minute")
        fig = px.histogram(df, x='marks_per_minute', nbins=30,
                          title='Distribution of Marks per Minute')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Marks/Minute", f"{df['marks_per_minute'].mean():.4f}")
        with col2:
            st.metric("Best Efficiency", f"{df['marks_per_minute'].max():.4f}")
        
        st.subheader("2. Efficiency Ratio Analysis")
        top_efficient = df.nlargest(10, 'marks_per_minute')[
            ['Full Name', 'marks_per_minute', 'marks_obtained', 'time_taken_minutes']
        ]
        st.dataframe(top_efficient, use_container_width=True)
        
        st.subheader("3. Cost-Benefit Analysis")
        df['gain_loss_diff'] = df['marks_gained'] - df['marks_lost']
        # Remove rows with NaN values for the scatter plot
        df_valid = df.dropna(subset=['marks_gained', 'marks_lost', 'percentage'])
        fig = px.scatter(df_valid, x='marks_gained', y='marks_lost',
                        size='percentage', hover_data=['Full Name'],
                        title='Marks Gained vs Lost (size = percentage)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("4. Return on Time Investment")
        fig = px.scatter(df, x='time_taken_minutes', y='marks_obtained',
                        color='score_category',
                        title='Time vs Marks (colored by category)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("7. Full Attempt vs Selective Strategy")
        df['strategy'] = df['skip_rate'].apply(
            lambda x: 'Full Attempt' if x < 10 else 'Selective'
        )
        strategy_perf = df.groupby('strategy')['percentage'].mean().reset_index()
        fig = px.bar(strategy_perf, x='strategy', y='percentage',
                    title='Performance by Strategy Type')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 9: STATISTICAL ANALYSIS
    with tabs[8]:
        st.header("Statistical Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df['percentage'].mean():.2f}")
        with col2:
            st.metric("Median", f"{df['percentage'].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df['percentage'].std():.2f}")
        with col4:
            mode_val = df['percentage'].mode()
            st.metric("Mode", f"{mode_val.iloc[0]:.2f}" if len(mode_val) > 0 else "N/A")
        
        st.subheader("1-3. Distribution Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Descriptive Statistics**")
            st.dataframe(df['percentage'].describe(), use_container_width=True)
        
        with col2:
            skewness = df['percentage'].skew()
            kurtosis = df['percentage'].kurtosis()
            
            st.write("**Distribution Metrics**")
            dist_metrics = pd.DataFrame({
                'Metric': ['Skewness', 'Kurtosis'],
                'Value': [skewness, kurtosis]
            })
            st.dataframe(dist_metrics, use_container_width=True)
            
            if abs(skewness) < 0.5:
                st.info("Distribution is approximately symmetric")
            elif skewness > 0.5:
                st.warning("Distribution is right-skewed (positive skew)")
            else:
                st.warning("Distribution is left-skewed (negative skew)")
        
        st.subheader("4. Quartile Analysis")
        q1 = df['percentage'].quantile(0.25)
        q2 = df['percentage'].quantile(0.50)
        q3 = df['percentage'].quantile(0.75)
        iqr = q3 - q1
        
        quartile_df = pd.DataFrame({
            'Quartile': ['Q1 (25%)', 'Q2 (50% - Median)', 'Q3 (75%)', 'IQR'],
            'Value': [q1, q2, q3, iqr]
        })
        st.dataframe(quartile_df, use_container_width=True)
        
        fig = px.box(df, y='percentage', title='Box Plot with Quartiles')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("5. Z-Score Analysis (Outlier Detection)")
        df['z_score'] = np.abs(stats.zscore(df['percentage']))
        outliers = df[df['z_score'] > 2]
        
        st.write(f"**Students with |Z-score| > 2:** {len(outliers)}")
        if len(outliers) > 0:
            outlier_display = outliers[['Full Name', 'percentage', 'z_score']].sort_values('z_score', ascending=False)
            st.dataframe(outlier_display, use_container_width=True)
        
        fig = px.scatter(df, x=df.index, y='z_score',
                        hover_data=['Full Name', 'percentage'],
                        title='Z-Score Distribution (Outliers > 2)')
        fig.add_hline(y=2, line_dash="dash", line_color="red", 
                     annotation_text="Outlier Threshold")
        fig.add_hline(y=-2, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("6. Statistical Significance Testing")
        
        # Department comparison if available
        dept_col = None
        if 'Department' in df.columns:
            dept_col = 'Department'
        elif 'College Department' in df.columns:
            dept_col = 'College Department'
        
        if dept_col and df[dept_col].nunique() > 1:
            st.write("**ANOVA Test: Performance across Departments**")
            
            groups = [group['percentage'].values for name, group in df.groupby(dept_col)]
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                
                anova_results = pd.DataFrame({
                    'Test': ['ANOVA'],
                    'F-Statistic': [f_stat],
                    'P-Value': [p_value],
                    'Significant (α=0.05)': ['Yes' if p_value < 0.05 else 'No']
                })
                st.dataframe(anova_results, use_container_width=True)
                
                if p_value < 0.05:
                    st.success("✅ Significant difference found between departments (p < 0.05)")
                else:
                    st.info("ℹ️ No significant difference between departments (p ≥ 0.05)")
        
        # College comparison if available
        if 'College Name' in df.columns and df['College Name'].nunique() > 1:
            st.write("**ANOVA Test: Performance across Colleges**")
            
            college_groups = [group['percentage'].values for name, group in df.groupby('College Name')]
            
            if len(college_groups) >= 2:
                f_stat_college, p_value_college = stats.f_oneway(*college_groups)
                
                anova_college = pd.DataFrame({
                    'Test': ['ANOVA'],
                    'F-Statistic': [f_stat_college],
                    'P-Value': [p_value_college],
                    'Significant (α=0.05)': ['Yes' if p_value_college < 0.05 else 'No']
                })
                st.dataframe(anova_college, use_container_width=True)
                
                if p_value_college < 0.05:
                    st.success("✅ Significant difference found between colleges (p < 0.05)")
                else:
                    st.info("ℹ️ No significant difference between colleges (p ≥ 0.05)")
        
        st.subheader("Normal Distribution Test")
        statistic, p_value_norm = stats.shapiro(df['percentage'].sample(min(5000, len(df))))
        
        normality_df = pd.DataFrame({
            'Test': ['Shapiro-Wilk'],
            'Statistic': [statistic],
            'P-Value': [p_value_norm],
            'Normal Distribution': ['Yes' if p_value_norm > 0.05 else 'No']
        })
        st.dataframe(normality_df, use_container_width=True)
    
    # Additional Analysis Section
    st.markdown("---")
    st.header("📥 Download Analysis Report")
    
    # Create summary statistics
    summary_stats = {
        'Total Students': len(df),
        'Average Score': df['marks_obtained'].mean(),
        'Average Percentage': df['percentage'].mean(),
        'Pass Rate (>40%)': (df['percentage'] > 40).sum() / len(df) * 100,
        'Perfect Scores': (df['percentage'] == 100).sum(),
        'Average Time (min)': df['time_taken_minutes'].mean(),
        'Avg Focus Changes': df['attempt_focus_change_count'].mean() if 'attempt_focus_change_count' in df.columns else 0,
        'Standard Deviation': df['percentage'].std()
    }
    
    summary_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        st.dataframe(summary_df, use_container_width=True)
    
    with col2:
        st.subheader("Quick Actions")
        
        # Download button for processed data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Processed Data",
            data=csv,
            file_name="student_analysis_processed.csv",
            mime="text/csv"
        )
        
        # Download summary
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to Student Assessment Analysis Dashboard! 👋
    
    This comprehensive tool analyzes student performance data across **100+ metrics** including:
    
    ### 📊 Analysis Categories:
    
    1. **Performance Analysis (20 metrics)**
       - Score distributions, pass/fail rates, top/low performers
       - Marks gained vs lost, attempt patterns, accuracy metrics
    
    2. **Time-Based Analysis (15 metrics)**
       - Time vs performance correlation, optimal time ranges
       - Speed vs accuracy, peak performance hours
    
    3. **Demographic & Institutional (12 metrics)**
       - College-wise, department-wise, section-wise comparisons
       - Institutional benchmarking
    
    4. **Behavioral & Engagement (13 metrics)**
       - Focus change analysis, test completion patterns
       - Risk-taking behavior, strategic answering
    
    5. **Technical Section Analysis (8 metrics)**
       - MCQ-specific performance and time allocation
    
    6. **Correlation Analysis (10 metrics)**
       - Multi-variable correlation studies
    
    7. **Benchmarking (8 metrics)**
       - Rankings, percentiles, peer comparisons
    
    8. **Efficiency Metrics (8 metrics)**
       - Marks per minute, ROI on time, strategy effectiveness
    
    9. **Statistical Analysis (6 metrics)**
       - Distribution tests, ANOVA, outlier detection
    
    ### 🚀 Getting Started:
    
    1. Upload your student assessment CSV file using the uploader above
    2. Navigate through the tabs to explore different analysis categories
    3. Download processed data and reports for further use
    
    ### 📋 Required CSV Format:
    
    Your CSV should contain columns like:
    - Student information (Full Name, Email, College, Department, Section)
    - Score metrics (marks_obtained, percentage, correct_count, incorrect_count)
    - Time metrics (total_time_taken, start_time, submit_time)
    - Engagement metrics (attempt_focus_change_count)
    - Technical section specific columns (optional)
    
    **Upload your CSV file to begin the analysis!**
    """)
    
    # Sample data structure
    with st.expander("📖 View Sample Data Structure"):
        st.code("""
Test,Full Name,Email,Phone Number,College Name,max_test_score,total_time_taken,
percentage,correct_count,incorrect_count,marks_gained,marks_lost,marks_obtained,
marks_skipped,start_time,submit_time,attempt_focus_change_count,Department,Section
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Student Assessment Analysis Dashboard v1.0 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)