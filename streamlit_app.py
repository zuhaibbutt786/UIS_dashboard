import streamlit as st
import pandas as pd
import plotly.express as px
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression

# --- ADVANCED ANALYTICS ENGINE ---
def parse_full_marks_report(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Metadata Extraction
    course_info = soup.find('small', class_='rptdata', string=re.compile(r'CS-|DS-'))
    course_name = course_info.text.strip() if course_info else "Academic Course"
    
    rows = soup.find_all('tr')
    assessment_names, total_marks, data = [], [], []

    # 2. Identify Headers and Totals (GIFT UIS Structure)
    for row in rows:
        text = row.get_text()
        if 'Q1' in text and 'Mid1' in text:
            assessment_names = [small.text.strip() for small in row.find_all('small') if small.text.strip()]
        
        if 'RollNo' in text and 'Name' in text:
            header_cells = row.find_all(['th', 'td'])
            for cell in header_cells:
                val = cell.get_text().strip()
                if val.isdigit():
                    total_marks.append(float(val))

    # 3. Process Student Rows
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 5: continue
        
        roll_text = cols[1].get_text().strip()
        if re.match(r'^\d{9}$', roll_text): # Detects GIFT 9-digit Roll Number
            roll_no, student_name = roll_text, cols[2].get_text().strip()
            marks_cols = cols[3:]
            for i, mark_td in enumerate(marks_cols):
                if i >= len(assessment_names): break
                comp_name = assessment_names[i]
                total = total_marks[i] if i < len(total_marks) else 10.0
                raw_val = mark_td.get_text().strip().upper()
                
                if raw_val in ['A', '', 'ABSENT']:
                    obtained, status = 0.0, "Absent"
                else:
                    try:
                        clean_val = re.sub(r'[^0-9.]', '', raw_val)
                        obtained = float(clean_val) if clean_val else 0.0
                        status = "Present"
                    except ValueError:
                        obtained, status = 0.0, "Error"

                data.append({
                    "Roll No": roll_no, "Name": student_name, "Assessment": comp_name,
                    "Obtained": obtained, "Total": total, "Status": status,
                    "Percentage": (obtained / total * 100) if total > 0 else 0
                })
    return pd.DataFrame(data), course_name

# --- UI SETUP ---
st.set_page_config(page_title="GIFT Strategic Analytics", layout="wide", page_icon="🎓")

st.title("🎓 Faculty Strategic Intelligence Dashboard")
st.markdown("Comprehensive Descriptive, Diagnostic, Predictive, and Prescriptive Analytics")

with st.sidebar:
    st.header("📥 Data Input")
    raw_data = st.text_area("Paste HTML Source Code:", height=400)
    st.info("Tip: Right-click UIS -> View Page Source -> Copy All -> Paste here.")

if raw_data:
    try:
        df, course_name = parse_full_marks_report(raw_data)
        
        if not df.empty:
            # --- DATA AGGREGATION ---
            student_stats = df.groupby(['Roll No', 'Name']).agg({
                'Percentage': 'mean',
                'Status': lambda x: (x == 'Present').sum(),
                'Obtained': 'sum',
                'Total': 'sum'
            }).reset_index()
            student_stats['Overall %'] = (student_stats['Obtained'] / student_stats['Total']) * 100

            # --- SECTION 1: STRATEGIC KPIs ---
            st.header(f"📌 Class Health Scorecard: {course_name}")
            k1, k2, k3, k4 = st.columns(4)
            
            avg_perf = student_stats['Overall %'].mean()
            attendance_rate = (df['Status'] == "Present").mean() * 100
            at_risk = len(student_stats[student_stats['Overall %'] < 50])
            total_students = len(student_stats)

            k1.metric("Class Average", f"{avg_perf:.1f}%")
            k2.metric("Engagement (Attendance)", f"{attendance_rate:.1f}%")
            k3.metric("Critical Risk (Fail)", at_risk, delta=f"{(at_risk/total_students)*100:.1f}% of class", delta_color="inverse")
            k4.metric("Total Enrolled", total_students)

            # --- ANALYTICS TABS ---
            tabs = st.tabs(["📊 Descriptive", "🔍 Diagnostic", "🔮 Predictive", "💊 Prescriptive", "📋 Data Explorer"])

            # 1. DESCRIPTIVE
            with tabs[0]:
                col_top, col_dist = st.columns([1, 2])
                with col_top:
                    st.subheader("🏆 Top 5 Elite Performers")
                    top_5 = student_stats.nlargest(5, 'Overall %')[['Roll No', 'Name', 'Overall %']]
                    st.dataframe(top_5, hide_index=True)
                
                with col_dist:
                    st.subheader("Assessment Performance Spread")
                    fig_box = px.box(df, x="Assessment", y="Percentage", color="Assessment", points="all", hover_data=["Name"])
                    st.plotly_chart(fig_box, use_container_width=True)

            # 2. DIAGNOSTIC
            with tabs[1]:
                st.subheader("Root Cause & Consistency Analysis")
                pivot_df = df.pivot(index='Roll No', columns='Assessment', values='Percentage')
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Assessment Correlation Matrix**")
                    st.plotly_chart(px.imshow(pivot_df.corr(), text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
                with c2:
                    st.write("**Most Inconsistent Students (High Variance)**")
                    variance = pivot_df.std(axis=1).sort_values(ascending=False).head(10)
                    st.bar_chart(variance)

            # 3. PREDICTIVE
            with tabs[2]:
                st.subheader("Final Result Forecasting")
                X = student_stats['Overall %'].values.reshape(-1, 1)
                y = X * 1.05 # Assume 5% growth potential
                model = LinearRegression().fit(X, y)
                student_stats['Projected'] = model.predict(X).clip(0, 100)

                fig_pred = px.scatter(student_stats, x="Overall %", y="Projected", hover_name="Name", 
                                     trendline="ols", title="Current Standing vs. Forecasted Outcome")
                st.plotly_chart(fig_pred, use_container_width=True)

            # 4. PRESCRIPTIVE (NEW)
            with tabs[3]:
                st.subheader("🩺 Academic Prescriptions (Action Plan)")
                p1, p2 = st.columns(2)
                
                with p1:
                    st.error("🚨 ACTION: Remedial Intervention")
                    fail_list = student_stats[student_stats['Overall %'] < 50].sort_values('Overall %')
                    st.write("These students are currently failing. Recommendation: Mandatory Review Session.")
                    st.dataframe(fail_list[['Roll No', 'Name', 'Overall %']], hide_index=True)
                
                with p2:
                    st.success("💎 ACTION: Peer Mentorship Program")
                    mentors = student_stats.nlargest(3, 'Overall %')
                    st.write("Top potential student mentors for peer-assisted learning:")
                    for idx, row in mentors.iterrows():
                        st.write(f"- **{row['Name']}** (Score: {row['Overall %']:.1f}%)")

            # 5. DATA EXPLORER
            with tabs[4]:
                st.subheader("Raw Ledger")
                raw_pivot = df.pivot(index=['Roll No', 'Name'], columns='Assessment', values='Obtained').reset_index()
                st.dataframe(raw_pivot, use_container_width=True)
                st.download_button("Export to CSV", raw_pivot.to_csv(index=False).encode('utf-8'), "marks.csv", "text/csv")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("Please paste the UIS source code in the sidebar to generate the full intelligence suite.")
