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
    
    # --- FIXED SECTION ---
    # Extract Assessment Names (Q1, Mid1, etc.)
    assessment_names = [small.text.strip() for small in rows[1].find_all('small') if small.text.strip()]
    
    # Extract Totals safely: Only keep it if it's a digit/number
    total_marks = []
    for small in rows[2].find_all('small'):
        val = small.text.strip()
        try:
            total_marks.append(float(val))
        except ValueError:
            continue # Skip non-numeric values like empty strings or labels
    
    # Ensure our mapping only includes assessments that have a corresponding total
    min_len = min(len(assessment_names), len(total_marks))
    assessment_names = assessment_names[:min_len]
    total_marks = total_marks[:min_len]
    assessment_map = dict(zip(assessment_names, total_marks))
    # ----------------------
    
    data = []
    for row in rows[3:]:
        cols = row.find_all('td')
        if len(cols) < 3: continue
        roll_no = cols[1].text.strip()
        student_name = cols[2].text.strip()
        
        # Start looking for marks from column index 3 onwards
        for i, mark_td in enumerate(cols[3:]):
            if i >= len(assessment_names): break
            
            comp_name = assessment_names[i]
            total = assessment_map[comp_name]
            raw_val = mark_td.text.strip().upper() # Handle 'A' or 'a'
            
            if raw_val == 'A' or raw_val == '':
                obtained = 0.0
                status = "Absent"
            else:
                try:
                    # Remove any non-numeric noise (like extra dots)
                    clean_val = re.sub(r'[^0-9.]', '', raw_val)
                    obtained = float(clean_val) if clean_val else 0.0
                    status = "Present"
                except ValueError:
                    obtained = 0.0
                    status = "Error"

            data.append({
                "Roll No": roll_no, "Name": student_name, "Assessment": comp_name,
                "Obtained": obtained, "Total": total, "Status": status,
                "Percentage": (obtained/total*100) if total > 0 else 0
            })
            
    return pd.DataFrame(data), course_name

# --- UI SETUP ---
# --- DASHBOARD LOGIC ---
if raw_data:
    try:
        df, course_name = parse_full_marks_report(raw_data)
        
        if df.empty:
            st.error("No student data found. Please check if the HTML pasted is the correct 'Exam Components Report' table.")
        else:
            # Create Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "📋 Raw Data"])

            # --- 1. DESCRIPTIVE ANALYTICS ---
            with tab1:
                st.header(f"Course Analysis: {course_name}")
                
                # Verify columns exist before calculating
                required_cols = ['Status', 'Percentage', 'Name', 'Obtained', 'Assessment']
                if all(col in df.columns for col in required_cols):
                    col1, col2, col3 = st.columns(3)
                    
                    present_mask = df['Status'] == "Present"
                    # Handle case where no one is marked 'Present' yet
                    if present_mask.any():
                        avg_perf = df[present_mask]['Percentage'].mean()
                        attendance_rate = (present_mask).mean() * 100
                        top_student = df.loc[df[present_mask]['Obtained'].idxmax(), 'Name']
                        
                        col1.metric("Course Average", f"{avg_perf:.1f}%")
                        col2.metric("Avg. Attendance", f"{attendance_rate:.1f}%")
                        col3.metric("Top Performer", top_student)
                        
                        fig_box = px.box(df, x="Assessment", y="Percentage", color="Assessment", 
                                         title="Score Distribution per Assessment")
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.warning("No students are marked as 'Present' in this data.")
                else:
                    st.error(f"Data Schema Error: Missing columns. Found: {list(df.columns)}")

            # ... Rest of your tab logic ...

    except Exception as e:
        st.error(f"Critical System Error: {e}")
