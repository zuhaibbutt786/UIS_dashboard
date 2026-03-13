import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.linear_model import LinearRegression

# --- DATA PARSING ENGINE ---
def parse_full_marks_report(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Metadata Extraction
    course_info = soup.find('small', class_='rptdata', string=re.compile(r'CS-|DS-'))
    course_name = course_info.text.strip() if course_info else "Unknown Course"
    
    rows = soup.find_all('tr')
    if len(rows) < 4:
        return pd.DataFrame(), course_name

    # 2. Extract Assessment Labels (Q1, Mid1, etc.)
    assessment_names = [small.text.strip() for small in rows[1].find_all('small') if small.text.strip()]
    
    # 3. Extract Totals safely
    total_marks = []
    for small in rows[2].find_all('small'):
        val = small.text.strip()
        try:
            total_marks.append(float(val))
        except ValueError:
            continue
    
    # Align headers and totals
    min_len = min(len(assessment_names), len(total_marks))
    assessment_names = assessment_names[:min_len]
    total_marks = total_marks[:min_len]
    assessment_map = dict(zip(assessment_names, total_marks))
    
    data = []
    # Student data starts from the 4th row
    for row in rows[3:]:
        cols = row.find_all('td')
        if len(cols) < 3: continue
        
        roll_no = cols[1].text.strip()
        student_name = cols[2].text.strip()
        
        for i, mark_td in enumerate(cols[3:]):
            if i >= len(assessment_names): break
            
            comp_name = assessment_names[i]
            total = assessment_map[comp_name]
            raw_val = mark_td.text.strip().upper()
            
            # Absentee & Empty logic
            if raw_val in ['A', '', 'ABSENT']:
                obtained = 0.0
                status = "Absent"
            else:
                try:
                    # Clean non-numeric characters except decimal
                    clean_val = re.sub(r'[^0-9.]', '', raw_val)
                    obtained = float(clean_val) if clean_val else 0.0
                    status = "Present"
                except ValueError:
                    obtained = 0.0
                    status = "Error"

            data.append({
                "Roll No": roll_no,
                "Name": student_name,
                "Assessment": comp_name,
                "Obtained": obtained,
                "Total": total,
                "Percentage": (obtained / total * 100) if total > 0 else 0,
                "Status": status
            })
            
    return pd.DataFrame(data), course_name

# --- UI SETUP ---
st.set_page_config(page_title="GIFT UIS Analytics", layout="wide", page_icon="📊")

st.title("🚀 GIFT UIS: Advanced Analytics Suite")
st.markdown("Convert raw portal data into Descriptive, Diagnostic, and Predictive insights.")

with st.sidebar:
    st.header("Data Entry")
    raw_data = st.text_area("Paste 'Exam Components Detail' HTML source here:", height=400)
    st.info("Tip: Right-click the UIS page -> View Page Source -> Copy All.")

# --- DASHBOARD LOGIC ---
if raw_data:
    try:
        df, course_name = parse_full_marks_report(raw_data)
        
        if df.empty:
            st.warning("No valid student data found. Please ensure you are pasting the full table source.")
        else:
            st.header(f"Analysis for: {course_name}")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Descriptive (What)", 
                "🔍 Diagnostic (Why)", 
                "🔮 Predictive (Next)", 
                "📋 Data Explorer"
            ])

            # --- TAB 1: DESCRIPTIVE ---
            with tab1:
                p_mask = df['Status'] == "Present"
                if p_mask.any():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Class Average", f"{df[p_mask]['Percentage'].mean():.1f}%")
                    m2.metric("Attendance", f"{(p_mask.mean()*100):.1f}%")
                    m3.metric("Top Score", f"{df['Percentage'].max():.1f}%")
                    m4.metric("Failure Risk (<50%)", len(df.groupby('Roll No')['Percentage'].mean().loc[lambda x: x < 50]))

                    st.subheader("Performance Distribution by Assessment")
                    fig_box = px.box(df, x="Assessment", y="Percentage", color="Assessment", points="all", hover_data=["Name"])
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.error("No 'Present' data available to describe.")

            # --- TAB 2: DIAGNOSTIC ---
            with tab2:
                st.subheader("Correlation & Consistency Analysis")
                pivot_df = df.pivot(index='Roll No', columns='Assessment', values='Percentage')
                
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.write("**Assessment Inter-dependence**")
                    corr = pivot_df.corr()
                    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='Blues'), use_container_width=True)
                    st.caption("High correlation indicates that these assessments tested similar student skills.")
                
                with col_b:
                    st.write("**Most Inconsistent Students**")
                    st.caption("Measured by Standard Deviation of scores across all exams.")
                    inconsistency = pivot_df.std(axis=1).sort_values(ascending=False).head(10)
                    st.bar_chart(inconsistency)

            # --- TAB 3: PREDICTIVE ---
            with tab3:
                st.subheader("Final Grade Forecasting")
                
                # Student-level average
                student_stats = df.groupby(['Roll No', 'Name'])['Percentage'].mean().reset_index()
                
                # Linear Regression: Using current average to predict a 'Target'
                # We assume a 5% improvement for the 'Predicted' potential
                X = student_stats['Percentage'].values.reshape(-1, 1)
                y = X * 1.05 # Baseline prediction model
                
                model = LinearRegression().fit(X, y)
                student_stats['Predicted_Final'] = model.predict(X).clip(0, 100)

                fig_pred = px.scatter(student_stats, x="Percentage", y="Predicted_Final", 
                                     hover_data=["Name"], trendline="ols",
                                     title="Current Average vs. Projected Final Grade")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.write("**Projected Risk List**")
                st.dataframe(student_stats[student_stats['Percentage'] < 50].sort_values('Percentage'))

            # --- TAB 4: DATA EXPLORER ---
            with tab4:
                st.subheader("Raw Pivot Table")
                wide_df = df.pivot(index=['Roll No', 'Name'], columns='Assessment', values='Obtained').reset_index()
                st.dataframe(wide_df, use_container_width=True)
                
                csv = wide_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Cleaned CSV", csv, "marks_report.csv", "text/csv")

    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.info("Check if the HTML format has changed. Ensure all columns have numerical totals.")
else:
    st.info("Waiting for data. Please paste the HTML source in the sidebar to generate analytics.")
