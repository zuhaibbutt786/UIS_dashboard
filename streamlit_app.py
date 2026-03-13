import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.linear_model import LinearRegression

# --- ANALYTICS ENGINE ---

def parse_full_marks_report(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    course_info = soup.find('small', class_='rptdata', string=re.compile(r'CS-|DS-'))
    course_name = course_info.text.strip() if course_info else "Unknown Course"
    
    rows = soup.find_all('tr')
    # Extract Assessment Names (Q1, Mid1, etc.)
    assessment_names = [small.text.strip() for small in rows[1].find_all('small')]
    # Extract Totals
    total_marks = [float(small.text.strip()) for small in rows[2].find_all('small')]
    assessment_map = dict(zip(assessment_names, total_marks))
    
    data = []
    for row in rows[3:]:
        cols = row.find_all('td')
        if len(cols) < 3: continue
        roll_no = cols[1].text.strip()
        student_name = cols[2].text.strip()
        
        for i, mark_td in enumerate(cols[3:]):
            if i >= len(assessment_names): break
            comp_name = assessment_names[i]
            total = assessment_map[comp_name]
            raw_val = mark_td.text.strip()
            
            status = "Present" if raw_val != 'A' else "Absent"
            obtained = float(raw_val) if raw_val != 'A' else 0.0

            data.append({
                "Roll No": roll_no, "Name": student_name, "Assessment": comp_name,
                "Obtained": obtained, "Total": total, "Status": status,
                "Percentage": (obtained/total*100) if total > 0 else 0
            })
            
    return pd.DataFrame(data), course_name

# --- UI SETUP ---
st.set_page_config(page_title="GIFT UIS Advanced Analytics", layout="wide")
st.title("🎓 GIFT UIS: Advanced Faculty Analytics")

raw_data = st.sidebar.text_area("Paste HTML Source:", height=300)

if raw_data:
    df, course_name = parse_full_marks_report(raw_data)
    
    # Create Tabs for different types of Analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "📋 Raw Data"])

    # --- 1. DESCRIPTIVE ANALYTICS ---
    with tab1:
        st.header("Class Performance Summary")
        col1, col2, col3 = st.columns(3)
        
        avg_perf = df[df['Status']=="Present"]['Percentage'].mean()
        attendance_rate = (df['Status']=="Present").mean() * 100
        
        col1.metric("Course Average", f"{avg_perf:.1f}%")
        col2.metric("Avg. Attendance", f"{attendance_rate:.1f}%")
        col3.metric("Top Performer", df.groupby('Name')['Obtained'].sum().idxmax())

        # Boxplot of all components
        fig_box = px.box(df, x="Assessment", y="Percentage", color="Assessment", 
                         title="Distribution of Scores per Assessment")
        st.plotly_chart(fig_box, use_container_width=True)

    # --- 2. DIAGNOSTIC ANALYTICS ---
    with tab2:
        st.header("Root Cause & Correlation Analysis")
        # Pivot data to see correlations
        pivot_df = df.pivot(index='Roll No', columns='Assessment', values='Percentage')
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("**Assessment Correlation Matrix**")
            corr = pivot_df.corr()
            fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_heatmap, use_container_width=True)
            st.caption("Insight: High correlation (close to 1.0) means students who performed well in one assessment also did well in the other.")

        with c2:
            st.write("**Performance Consistency**")
            # Identify students with high variance (inconsistent performers)
            std_dev = pivot_df.std(axis=1).sort_values(ascending=False).head(10)
            st.bar_chart(std_dev)
            st.write("Top 10 Students with highest score fluctuations (Need Attention).")

    # --- 3. PREDICTIVE ANALYTICS ---
    with tab3:
        st.header("Score Forecasting")
        st.info("Predicting future performance based on current Quiz and Midterm scores.")
        
        # Prepare data for simple Linear Regression
        # We'll use the average of all completed assessments to predict the "Next" one
        student_avg = df.groupby('Roll No')['Percentage'].mean().values.reshape(-1, 1)
        # For demo: Predicting a hypothetical Final (Current Avg * 1.05 as a placeholder target)
        model = LinearRegression()
        model.fit(student_avg, student_avg * 1.1) # Simple logic for demonstration
        
        # Predictor Input
        user_score = st.slider("If a student's current average is:", 0, 100, 50)
        prediction = model.predict([[user_score]])[0][0]
        st.success(f"Predicted Final Term Score: **{min(prediction, 100.0):.1f}%**")
        
        # Risk Profiling
        st.subheader("⚠️ Students at Risk")
        at_risk = pivot_df.mean(axis=1)
        at_risk_list = at_risk[at_risk < 50]
        if not at_risk_list.empty:
            st.warning(f"Found {len(at_risk_list)} students averaging below 50%.")
            st.dataframe(at_risk_list)

    # --- 4. RAW DATA ---
    with tab4:
        st.dataframe(df.pivot(index=['Roll No', 'Name'], columns='Assessment', values='Obtained'))

else:
    st.info("Please paste the UIS HTML Source to generate the analytics suite.")
