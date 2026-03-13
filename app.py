import streamlit as st
import pandas as pd
import plotly.express as px
from bs4 import BeautifulSoup
import re

# --- DATA PARSING FUNCTIONS ---
def parse_academic_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Identify the Course (from the second HTML structure provided)
    course_info = soup.find('small', class_='rptdata', string=re.compile(r'CS-|DS-'))
    course_name = course_info.text if course_info else "Unknown Course"
    
    test_type_info = soup.find('td', string="Test Type:")
    test_label = "Assessment"
    if test_type_info:
        test_label = test_type_info.find_next('small').text

    # Extract Table
    table = soup.find('table', class_='tbl_sort')
    rows = table.find_all('tr')[1:] # Skip header
    
    data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 6:
            obtained = float(cols[3].text.strip())
            total = float(cols[4].text.strip())
            
            # Handle the -1 (Absent) logic
            status = "Present" if obtained != -1 else "Absent"
            actual_score = obtained if obtained != -1 else 0
            
            data.append({
                "Roll No": cols[1].text.strip(),
                "Name": cols[2].text.strip(),
                "Obtained": actual_score,
                "Total": total,
                "Percentage": (actual_score / total) * 100,
                "Status": status,
                "Course": course_name,
                "Assessment": test_label
            })
    return pd.DataFrame(data)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Academic Analytics Dashboard", layout="wide")

st.title("📊 Faculty Analytics Dashboard")
st.markdown("Paste your UI HTML or upload the saved report to generate live insights.")

# Sidebar for Data Input
with st.sidebar:
    st.header("Data Input")
    input_method = st.radio("Choose Input:", ["Paste HTML", "Upload File"])
    raw_data = ""
    
    if input_method == "Paste HTML":
        raw_data = st.text_area("Paste HTML Source here...", height=300)
    else:
        uploaded_file = st.file_uploader("Upload HTML report", type=['html', 'htm'])
        if uploaded_file:
            raw_data = uploaded_file.read().decode("utf-8")

# --- DASHBOARD LOGIC ---
if raw_data:
    try:
        df = parse_academic_data(raw_data)
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        avg_score = df[df['Status'] == "Present"]['Percentage'].mean()
        attendance = (df['Status'] == "Present").sum() / len(df) * 100
        
        col1.metric("Total Students", len(df))
        col2.metric("Class Average", f"{avg_score:.1f}%")
        col3.metric("Attendance", f"{attendance:.1f}%")
        col4.metric("Highest Mark", f"{df['Obtained'].max()}")

        # Visualization Row
        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Grade Distribution")
            fig_hist = px.histogram(df, x="Percentage", nbins=10, 
                                   color_discrete_sequence=['#3366CC'],
                                   labels={'Percentage': 'Grade Percentage (%)'})
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.subheader("Student Performance List")
            st.dataframe(df[['Roll No', 'Name', 'Obtained', 'Total', 'Status']], 
                         use_container_width=True, hide_index=True)

        # Detailed Analysis
        st.divider()
        st.subheader("Performance Breakdown")
        fig_box = px.box(df, y="Percentage", points="all", hover_data=["Name"])
        st.plotly_chart(fig_box, use_container_width=True)

    except Exception as e:
        st.error(f"Error parsing data: {e}. Please ensure you are pasting the 'Students Marks Sheet' HTML.")
else:
    st.info("Waiting for data... Please paste the HTML from your assessment view.")
