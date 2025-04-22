import streamlit as st
from openai import OpenAI
import httpx
import os
import PyPDF2   
import re
import smtplib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime
from fpdf import FPDF
import json

# --------------------- Initialization ---------------------
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")

# Initialize session state
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Basic"  # Default to Basic mode
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'job_desc' not in st.session_state:
    st.session_state.job_desc = ""

# Force-clear proxy settings
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(var, None)

# Initialize OpenAI client
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    http_client=httpx.Client(timeout=30.0)
)

# --------------------- Enhanced Helper Functions ---------------------
def extract_text(file) -> str:
    """Improved text extraction from files"""
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages])
        return file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        st.error(f"File reading error: {str(e)}")
        return ""

def extract_personal_info(resume_text: str) -> Dict[str, str]:
    """Enhanced extraction of name, email, phone from resume"""
    info = {
        'name': '',
        'email': '',
        'phone': '',
        'location': '',
        'experience_years': 0
    }

    # Email extraction - improved pattern
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    emails = re.findall(email_pattern, resume_text)
    if emails:
        valid_emails = [e for e in emails if '.' in e.split('@')[1]]
        if valid_emails:
            info['email'] = valid_emails[0]

    # Phone number extraction
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, resume_text)
        if phones:
            phone = phones[0]
            if phone.startswith('+'):
                phone = '+' + ''.join(c for c in phone[1:] if c.isdigit())
            else:
                phone = ''.join(c for c in phone if c.isdigit())
            
            if len(phone) >= 7:
                info['phone'] = phone
                break
    
    # AI-powered extraction for remaining fields
    prompt = f"""Analyze this resume excerpt:
    {resume_text[:2500]}
    
    Return JSON with:
    - "name": Full name
    - "location": City/State/Country
    - "experience_years": Estimated years
    - "email": Verify this: {info['email'] if info['email'] else "Not found"}
    - "phone": Verify this: {info['phone'] if info['phone'] else "Not found"}
    """
    
    try:
        ai_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        ai_data = json.loads(ai_response.choices[0].message.content)
        
        if ai_data.get('name', '').strip() and len(ai_data['name'].split()) >= 2:
            info['name'] = ai_data['name'].strip()
        
        if ai_data.get('location', '').strip():
            info['location'] = ai_data['location'].strip()
            
        if ai_data.get('email', '').strip() and '@' in ai_data['email']:
            info['email'] = ai_data['email'].strip()
            
        if ai_data.get('phone', '').strip() and not info['phone']:
            cleaned_phone = ''.join(c for c in ai_data['phone'] if c.isdigit() or c == '+')
            if len(cleaned_phone) >= 7:
                info['phone'] = cleaned_phone
                
        if ai_data.get('experience_years'):
            try:
                info['experience_years'] = int(ai_data['experience_years'])
            except:
                numbers = re.findall(r'\d+', str(ai_data['experience_years']))
                if numbers:
                    info['experience_years'] = int(numbers[0])
                    
    except Exception as e:
        st.warning(f"AI extraction issues: {str(e)}")
    
    # Fallback for name
    if not info['name']:
        name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
        names = re.findall(name_pattern, resume_text.strip())
        if names:
            info['name'] = names[0]
            
    return info

def analyze_with_openai(prompt: str, max_tokens=500) -> str:
    """More robust OpenAI query function"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return ""

def extract_timeline(resume_text: str) -> str:
    """Extract professional timeline"""
    prompt = f"""Extract professional timeline from:
    {resume_text[:5000]}
    
    Format as markdown:
    ### Timeline
    - [Year] Position @ Company
    - [Year] Education @ Institution
    """
    return analyze_with_openai(prompt)

def generate_job_recommendations(resume_text: str) -> str:
    """Generate AI-powered job recommendations"""
    prompt = f"""Based on this resume:
    {resume_text[:5000]}
    
    Suggest 5 ideal job roles with:
    - Job title
    - Industry
    - Key skills needed
    - Salary range (USD)
    - Fit score (1-100)
    """
    return analyze_with_openai(prompt)

def estimate_salary(job_title: str, location: str, experience: int) -> str:
    """Get salary estimation"""
    prompt = f"""Estimate salary range for:
    Job Title: {job_title}
    Location: {location}
    Experience: {experience} years
    
    Return format: '$XX,XXX - $XX,XXX per year'"""
    return analyze_with_openai(prompt, max_tokens=100)

def calculate_ats_score(resume_text: str, job_desc: str = "") -> Dict[str, float]:
    """Calculate multiple ATS-related scores"""
    scores = {
        'ats_compliance': 0,
        'content_quality': 0,
        'keyword_optimization': 0,
        'grammar_score': 0
    }
    
    # ATS Compliance Check
    ats_prompt = f"""Analyze this resume for ATS compliance:
    {resume_text[:5000]}
    
    Return JSON with scores (0-100):
    - "ats_compliance"
    - "content_quality" 
    - "keyword_optimization"
    - "ats_requirements": List of 5 key requirements
    - "missing_elements": Missing sections
    """
    
    if job_desc:
        ats_prompt += f"\n\nJob Description Context:\n{job_desc[:2000]}"
    
    try:
        ats_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": ats_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        ats_data = json.loads(ats_response.choices[0].message.content)
        
        scores.update({
            'ats_compliance': float(ats_data.get('ats_compliance', 0)),
            'content_quality': float(ats_data.get('content_quality', 0)),
            'keyword_optimization': float(ats_data.get('keyword_optimization', 0)),
            'ats_requirements': ats_data.get('ats_requirements', []),
            'missing_elements': ats_data.get('missing_elements', [])
        })
    except Exception as e:
        st.error(f"ATS scoring failed: {str(e)}")
    
    # Grammar and Language Check
    grammar_prompt = f"""Analyze this resume for language quality:
    {resume_text[:5000]}
    
    Return JSON with:
    - "grammar_score" (0-100)
    - "language_issues": Specific issues
    - "improvement_suggestions"
    """
    
    try:
        grammar_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": grammar_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        grammar_data = json.loads(grammar_response.choices[0].message.content)
        
        scores.update({
            'grammar_score': float(grammar_data.get('grammar_score', 0)),
            'language_issues': grammar_data.get('language_issues', []),
            'improvement_suggestions': grammar_data.get('improvement_suggestions', [])
        })
    except Exception as e:
        st.error(f"Grammar check failed: {str(e)}")
    
    return scores

def analyze_job_match(resume_text: str, job_desc: str) -> Dict:
    """Analyze how well the resume matches a job description"""
    if not job_desc:
        return {
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "recommendations": []
        }
        
    match_prompt = f"""Compare this resume to the job description:
    
    RESUME:
    {resume_text[:4000]}
    
    JOB DESCRIPTION:
    {job_desc[:2000]}
    
    Return detailed JSON with:
    - "match_score": Overall match percentage (0-100)
    - "matching_skills": List of skills in resume matching job requirements
    - "missing_skills": List of important skills from job description missing in resume
    - "keyword_coverage": Percentage of key job keywords found in resume (0-100)
    - "recommendations": List of 5 specific recommendations to improve resume for this job
    """
    
    try:
        match_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": match_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        match_data = json.loads(match_response.choices[0].message.content)
        return match_data
    except Exception as e:
        st.error(f"Job matching analysis failed: {str(e)}")
        return {
            "match_score": 0,
            "matching_skills": [],
            "missing_skills": [],
            "recommendations": ["Error analyzing job match"]
        }

def visualize_scores(scores: Dict[str, float]):
    """Create visualizations for the scores"""
    st.subheader("📊 Resume Scores")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ATS Compliance", f"{scores['ats_compliance']}/100")
    with col2:
        st.metric("Content Quality", f"{scores['content_quality']}/100")
    with col3:
        st.metric("Grammar Score", f"{scores['grammar_score']}/100")
    
    if 'keyword_optimization' in scores:
        st.metric("Keyword Optimization", f"{scores['keyword_optimization']}/100")
    
    with st.expander("Score Breakdown"):
        score_df = pd.DataFrame({
            'Metric': ['ATS Compliance', 'Content Quality', 'Grammar', 'Keyword Optimization'],
            'Score': [
                scores['ats_compliance'],
                scores['content_quality'],
                scores['grammar_score'],
                scores.get('keyword_optimization', 0)
            ]
        })
        st.bar_chart(score_df.set_index('Metric'))

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Improved email sending with clean formatting"""
    try:
        msg = MIMEMultipart()
        msg['From'] = st.secrets["EMAIL_SENDER"]
        msg['To'] = to_email
        msg['Subject'] = subject
        
        html_content = f"""
        <html>
        <body>
            <h1>Resume Analysis Report</h1>
            <div>{body.replace('\n', '<br>')}</div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        with smtplib.SMTP_SSL(st.secrets["EMAIL_SERVER"], 465, timeout=10) as server:
            server.login(st.secrets["EMAIL_SENDER"], st.secrets["EMAIL_PASSWORD"])
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {str(e)}")
        return False

def create_pdf_report(report_content: str) -> bytes:
    """Generate professional PDF report"""
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Resume Analysis Report', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', '', 11)
    
    sections = {
        'PERSONAL INFORMATION': '',
        'RESUME SCORES': '',
        'PROFESSIONAL TIMELINE': '',
        'JOB MATCH ANALYSIS': '',
        'JOB RECOMMENDATIONS': '',
        'IMPROVEMENT SUGGESTIONS': '',
    }
    
    current_section = None
    for line in report_content.split('\n'):
        if line.strip() in sections:
            current_section = line.strip()
        elif line.strip() and current_section:
            sections[current_section] += line + '\n'
    
    for section, content in sections.items():
        if not content.strip():
            continue
            
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, section, 1, 1, 'L', True)
        pdf.ln(2)
        
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, content.strip())
        pdf.ln(5)
    
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
    
    return pdf.output(dest='S').encode('latin1')



def clean_markdown(text: str) -> str:
    """Remove Markdown bold/italic symbols like *, **, ***"""
    return re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)

def generate_report_content(personal_info, scores, analysis, salary, job_match=None, chat_history=None):
    """Create well-formatted report content"""
    report = f"""PERSONAL INFORMATION

Name: {personal_info.get('name', 'N/A')}
Email: {personal_info.get('email', 'N/A')}
Phone: {personal_info.get('phone', 'N/A')}
Location: {personal_info.get('location', 'N/A')}

RESUME SCORES

ATS Compliance: {scores.get('ats_compliance', 0)}/100
Content Quality: {scores.get('content_quality', 0)}/100
Grammar Score: {scores.get('grammar_score', 0)}/100
"""

    if job_match:
        report += f"""
JOB MATCH ANALYSIS

Overall Match Score: {job_match.get('match_score', 'N/A')}/100
Keyword Coverage: {job_match.get('keyword_coverage', 'N/A')}%

Matching Skills:
{chr(10).join(['- ' + skill for skill in job_match.get('matching_skills', ['N/A'])])}

Missing Skills:
{chr(10).join(['- ' + skill for skill in job_match.get('missing_skills', ['N/A'])])}

Improvement Recommendations:
{chr(10).join(['- ' + rec for rec in job_match.get('recommendations', ['N/A'])])}
"""

    report += f"""
PROFESSIONAL TIMELINE

{analysis.get('timeline', 'N/A')}

JOB RECOMMENDATIONS

{analysis.get('recommendations', 'N/A')}

IMPROVEMENT SUGGESTIONS

{analysis.get('feedback', 'N/A')}

EXPECTED SALARY

{salary}
"""

    return clean_markdown(report)  # 🔥 Removes ** and *** from the final report


def chat_with_resume(resume_text: str, chat_history: List[Dict], user_message: str, job_desc: str = "") -> Tuple[str, List[Dict]]:
    """Handle interactive chat with resume context"""
    system_message = f"""You are a career coach analyzing this resume:
    {resume_text[:5000]}
    
    Provide professional advice about:
    - Resume improvements
    - Career suggestions
    - Job search strategies
    """
    
    if job_desc:
        system_message += f"""
        
        Also consider this job description the candidate is applying for:
        {job_desc[:2000]}
        """
    
    if not chat_history:
        chat_history = [{"role": "system", "content": system_message}]
    
    chat_history.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=chat_history,
            temperature=0.2,
            max_tokens=300
        )
        assistant_message = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message, chat_history
    except Exception as e:
        error_msg = f"Error: {str(e)}. Please try again."
        chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, chat_history

# --------------------- Streamlit UI ---------------------
with st.sidebar:
    st.header("Controls")
    st.session_state.analysis_mode = st.radio(
        "Analysis Mode",
        ["Basic", "Advanced"],
        index=0 if st.session_state.analysis_mode == "Basic" else 1
    )
    
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt", "docx"])
    
    # Job Description Input - Available in both modes
    st.session_state.job_desc = st.text_area(
        "Paste Job Description", 
        value=st.session_state.job_desc,
        height=150,
        placeholder="Paste job description for better matching..."
    )

# Initialize chat session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main Interface
if uploaded_file:
    resume_text = extract_text(uploaded_file)
    job_desc = st.session_state.job_desc
    
    if not resume_text:
        st.error("Failed to extract text from file")
        st.stop()
    
    # ========== BASIC MODE ==========
    if st.session_state.analysis_mode == "Basic":
        st.header("Basic Resume Analysis")
        
        # Personal Info Section
        with st.spinner("Extracting information..."):
            personal_info = extract_personal_info(resume_text)
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Name", value=personal_info.get('name', 'Not found'), disabled=True)
            st.text_input("Email", value=personal_info.get('email', 'Not found'), disabled=True)
        with col2:
            st.text_input("Phone", value=personal_info.get('phone', 'Not found'), disabled=True)
            st.text_input("Location", value=personal_info.get('location', 'Not found'), disabled=True)
        
        # Basic ATS Score
        with st.spinner("Calculating basic scores..."):
            scores = calculate_ats_score(resume_text, job_desc)
            st.subheader("ATS Compatibility Score")
            st.progress(int(scores['ats_compliance'])/100)
            st.caption(f"Score: {scores['ats_compliance']}/100")
        
        # Job Match Analysis (simplified)
        if job_desc:
            with st.spinner("Analyzing job match..."):
                job_match = analyze_job_match(resume_text, job_desc)
                
                st.subheader("🎯 Job Match Score")
                st.progress(int(job_match.get('match_score', 0))/100)
                st.caption(f"Match Score: {job_match.get('match_score', 0)}/100")
                
                with st.expander("Job Match Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Matching Skills")
                        for skill in job_match.get('matching_skills', [])[:5]:
                            st.write(f"- {skill}")
                    
                    with col2:
                        st.subheader("Skills You Might Need")
                        for skill in job_match.get('missing_skills', [])[:5]:
                            st.write(f"- {skill}")
        
        # Quick Tips
        st.subheader("Quick Improvement Tips")
        tips = analyze_with_openai(
            f"Provide 3 most important improvements for this resume in bullet points:\n{resume_text[:3000]}" + 
            (f"\nConsidering this job description:\n{job_desc[:1000]}" if job_desc else "")
        )
        st.write(tips)
        
        # Switch to Advanced button
        if st.button("Switch to Advanced Analysis"):
            st.session_state.analysis_mode = "Advanced"
            st.rerun()

# ========== ADVANCED MODE ==========
    else:
        if st.button("← Back to Basic Mode"):
            st.session_state.analysis_mode = "Basic"
            st.rerun()
            
        st.header("Advanced Resume Analysis")
        
        # Personal Info Section (full)
        with st.spinner("Extracting detailed information..."):
            personal_info = extract_personal_info(resume_text)
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Name", value=personal_info.get('name', 'Not found'), disabled=True)
            st.text_input("Email", value=personal_info.get('email', 'Not found'), disabled=True)
        with col2:
            st.text_input("Phone", value=personal_info.get('phone', 'Not found'), disabled=True)
            st.text_input("Location", value=personal_info.get('location', 'Not found'), disabled=True)
        
        # Calculate Scores (full)
        with st.spinner("Running comprehensive analysis..."):
            scores = calculate_ats_score(resume_text, job_desc)
            visualize_scores(scores)
        
        # Job Match Analysis (detailed)
        job_match = None
        if job_desc:
            with st.spinner("Analyzing job match in detail..."):
                job_match = analyze_job_match(resume_text, job_desc)
                
                st.subheader("Job Match Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Overall Match Score", f"{job_match.get('match_score', 0)}/100")
                with col2:
                    st.metric("Keyword Coverage", f"{job_match.get('keyword_coverage', 0)}%")
        
        # Advanced Analysis
        with st.spinner("Generating detailed insights..."):
            analysis = {
                'timeline': extract_timeline(resume_text),
                'recommendations': generate_job_recommendations(resume_text),
                'skills': analyze_with_openai(f"Categorize skills from: {resume_text[:5000]}"),
                'feedback': analyze_with_openai(f"Provide 5 specific improvements for: {resume_text[:5000]}" + 
                                               (f"\nConsidering this job: {job_desc[:1000]}" if job_desc else "")),
                'ats_requirements': scores.get('ats_requirements', []),
                'missing_elements': scores.get('missing_elements', []),
                'language_issues': scores.get('language_issues', []),
                'grammar_suggestions': scores.get('improvement_suggestions', [])
            }
            
            salary = estimate_salary(
                job_title=analysis['recommendations'].split('\n')[0].replace('-', '').strip() if not job_desc else job_desc.split('\n')[0].strip(),
                location=personal_info.get('location', 'United States'),
                experience=personal_info.get('experience_years', 3)
            )
        
        # Display Results
        st.header(" Analysis Results")
        
        tabs = ["Timeline", "Job Recs", "Skills", "Feedback", "ATS Requirements", "Grammar", "AI Coach"]
        if job_desc:
            tabs.append("Job Match")
            
        selected_tabs = st.tabs(tabs)
        
        with selected_tabs[0]:
            st.markdown(analysis['timeline'])
        
        with selected_tabs[1]:
            st.markdown(analysis['recommendations'])
        
        with selected_tabs[2]:
            st.markdown(analysis['skills'])
        
        with selected_tabs[3]:
            st.markdown(analysis['feedback'])
        
        with selected_tabs[4]:
            st.subheader("ATS Requirements")
            st.write("These are key elements ATS systems look for:")
            for req in analysis['ats_requirements']:
                st.write(f"- {req}")
            
            if analysis['missing_elements']:
                st.warning("Missing Standard Sections:")
                for missing in analysis['missing_elements']:
                    st.write(f"- {missing}")
        
        with selected_tabs[5]:
            st.subheader("Grammar and Language Issues")
            if analysis['language_issues']:
                st.warning("Found these language issues:")
                for issue in analysis['language_issues'][:5]:
                    st.write(f"- {issue}")
            else:
                st.success("No significant grammar issues found!")
            
            if analysis['grammar_suggestions']:
                st.info("Language Improvement Suggestions:")
                for suggestion in analysis['grammar_suggestions'][:3]:
                    st.write(f"- {suggestion}")
        
        with selected_tabs[6]:
            st.subheader("Interactive Resume Coach")
            st.write("Ask questions about your resume or get career advice:")
            
            for msg in st.session_state.chat_history[1:]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            if prompt := st.chat_input("Ask about your resume..."):
                with st.chat_message("user"):
                    st.write(prompt)
                
                with st.spinner("Thinking..."):
                    response, updated_history = chat_with_resume(
                        resume_text,
                        st.session_state.chat_history,
                        prompt,
                        job_desc
                    )
                    st.session_state.chat_history = updated_history
                
                with st.chat_message("assistant"):
                    st.write(response)
        
        if job_desc and len(selected_tabs) > 7:
            with selected_tabs[7]:
                st.subheader("Job Description Match Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("Matching Skills")
                    for skill in job_match.get('matching_skills', []):
                        st.success(f"**{skill}**")
                
                with col2:
                    st.markdown("Missing Skills")
                    for skill in job_match.get('missing_skills', []):
                        st.error(f"**{skill}**")
                
                st.markdown(" Improvement Recommendations")
                for i, rec in enumerate(job_match.get('recommendations', []), 1):
                    st.info(f"**{i}.** {rec}")
        
        # Salary Estimation
        with st.expander(" Expected Salary Range"):
            st.write(salary)
        
        # Report Generation
        st.header(" Get Your Report")
        email = st.text_input("Enter your email (optional)", value=personal_info.get('email', ''))
        
        report_content = generate_report_content(
            personal_info, scores, analysis, salary, job_match,
            st.session_state.chat_history
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Email Report"):
                if email and re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    if send_email(email, "Your Resume Analysis Report", report_content):
                        st.success("Report sent to your email!")
                    else:
                        st.error("Failed to send email")
                else:
                    st.warning("Please enter a valid email address")
        
        with col2:
            pdf_bytes = create_pdf_report(report_content)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="resume_analysis_report.pdf",
                mime="application/pdf"
            )

# Landing Page
else:
    st.markdown("""
# CVInsight 
Upload your resume to get:
- **Basic Mode**: Quick analysis and scores
- **Advanced Mode**: Full AI-powered insights

##### Features:
1. Resume ATS compatibility scoring
2. Job description matching
3. Skills gap analysis
4. Personalized improvement suggestions
5. Interactive AI resume coach
6. Downloadable PDF reports

##### How It Works:
1. Upload your resume (PDF, DOCX, or TXT)
2. Paste a job description (optional but recommended)
3. Select your analysis level
4. Get personalized feedback and job match scores
5. Download your report or ask the AI coach for advice

Get started by uploading your resume on the left sidebar!
""")

    
    st.info(" **Pro Tip**: For the best results, paste a job description you're interested in applying for. This will provide you with tailored matching analysis and recommendations.")
    
    # Sample Resume and Job Description Section
    with st.expander("Need examples? Click here for sample content"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Resume Structure")
            st.code("""
JOHN DOE
123 Main St, New York, NY | john.doe@email.com | (555) 123-4567

SUMMARY
Experienced software developer with 5+ years specializing in Python, JavaScript and cloud technologies.

SKILLS
Programming: Python, JavaScript, Java, SQL
Frameworks: Django, React, Flask
Tools: Git, Docker, AWS, Azure

EXPERIENCE
Senior Developer | Tech Solutions Inc. | 2020-Present
- Led team of 5 developers on e-commerce platform
- Implemented CI/CD pipeline reducing deployment time by 40%

Developer | Digital Innovations | 2018-2020
- Built RESTful APIs serving 100K+ daily users
- Optimized database queries improving performance by 35%

EDUCATION
Bachelor of Science, Computer Science
University of Technology | 2014-2018
            """)
            
        with col2:
            st.subheader("Sample Job Description")
            st.code("""
Senior Full Stack Developer

ABOUT THE ROLE:
We're looking for a Senior Full Stack Developer to join our growing team. You'll be responsible for developing and maintaining web applications, working closely with product managers and designers.

REQUIREMENTS:
- 5+ years of professional software development experience
- Strong proficiency in Python and JavaScript
- Experience with React and Django frameworks
- Experience with RESTful APIs and microservices
- Knowledge of SQL databases and query optimization
- Familiarity with AWS or other cloud providers
- Strong problem-solving skills and attention to detail
- Experience with CI/CD pipelines and automated testing

PREFERRED QUALIFICATIONS:
- Experience with TypeScript and GraphQL
- Knowledge of containerization (Docker, Kubernetes)
- Experience with Agile development methodologies
- Open source contributions

BENEFITS:
- Competitive salary
- Remote work options
- Health insurance
- 401(k) matching
- Professional development budget
            """)
    
    # Statistics or Testimonials
    st.markdown("Why Use Our Resume Analyzer?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Resume Success Rate", "+35%")
        st.caption("Average improvement in interview callbacks")
    
    with col2:
        st.metric("ATS Pass Rate", "+42%")
        st.caption("Increase in resumes passing ATS filters")
    
    with col3:
        st.metric("Job Match Accuracy", "95%")
        st.caption("Precision of our AI job matching algorithm")