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
        # Filter out common false positives like 'name@company' in sentences
        valid_emails = [e for e in emails if '.' in e.split('@')[1]]
        if valid_emails:
            info['email'] = valid_emails[0]  # Take the first valid email

    # Improved phone number extraction
    # This pattern handles various formats: (123) 456-7890, 123-456-7890, +1 123 456 7890, etc.
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'  # Simple 10-digit
    ]
    
    # Try each pattern until we find a match
    for pattern in phone_patterns:
        phones = re.findall(pattern, resume_text)
        if phones:
            # Clean the phone number
            phone = phones[0]
            # Remove non-digit characters except the leading + if present
            if phone.startswith('+'):
                phone = '+' + ''.join(c for c in phone[1:] if c.isdigit())
            else:
                phone = ''.join(c for c in phone if c.isdigit())
            
            # Validate minimum length
            if len(phone) >= 7:
                info['phone'] = phone
                break
    
    # Extract name, location, and experience using AI with better prompting
    # First look for common resume header patterns
    header_text = resume_text[:1000]  # Examine just the top portion
    
    # Extract name and location using AI with better context
    prompt = f"""Analyze this resume excerpt and extract the following information:
    {resume_text[:2500]}
    
    Return ONLY a JSON object with these keys (no explanation):
    - "name": The full name of the person (first and last name)
    - "location": City/State/Country or address
    - "experience_years": Estimated total years of professional experience based on work history
    - "email": Extract or verify this email: {info['email'] if info['email'] else "Not found in initial scan"}
    - "phone": Extract or verify this phone: {info['phone'] if info['phone'] else "Not found in initial scan"}
    
    Be precise with name extraction - this is typically one of the first prominent elements in a resume.
    Look for formatting cues like larger text, bold formatting, or centered text at the top.
    """
    
    try:
        ai_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using the more capable model for better extraction
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        ai_text = ai_response.choices[0].message.content.strip()
        
        # Proper error handling for JSON parsing
        try:
            ai_data = json.loads(ai_text)
        except:
            # Fallback to eval if json.loads fails
            try:
                ai_data = eval(ai_text)
            except:
                ai_data = {}
                st.warning("AI extraction encountered an error parsing the results")
        
        # Only update if AI found valid information
        if ai_data.get('name', '').strip() and len(ai_data.get('name', '').split()) >= 2:
            info['name'] = ai_data['name'].strip()
        
        if ai_data.get('location', '').strip():
            info['location'] = ai_data['location'].strip()
            
        if ai_data.get('email', '').strip() and '@' in ai_data['email'] and '.' in ai_data['email'].split('@')[1]:
            info['email'] = ai_data['email'].strip()
            
        if ai_data.get('phone', '').strip() and not info['phone']:
            # Clean the phone and verify
            cleaned_phone = ''.join(c for c in ai_data['phone'] if c.isdigit() or c == '+')
            if len(cleaned_phone) >= 7:  # Minimum valid phone length
                info['phone'] = cleaned_phone
                
        if ai_data.get('experience_years'):
            try:
                info['experience_years'] = int(ai_data['experience_years'])
            except:
                # Handle case where experience might be returned as text
                exp_text = str(ai_data['experience_years'])
                numbers = re.findall(r'\d+', exp_text)
                if numbers:
                    info['experience_years'] = int(numbers[0])
                    
    except Exception as e:
        st.warning(f"AI extraction had some issues: {str(e)}")
    
    # If AI extraction failed to find a name, try a simple regex approach as fallback
    if not info['name']:
        # Look for patterns like "John Doe" at the beginning of the resume
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
    prompt = f"""Extract professional timeline in chronological order from:
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
    - Salary range (USD if possible)
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
    
    Return JSON with these scores (0-100):
    - "ats_compliance": How well formatted it is for ATS
    - "content_quality": Quality of content and structure
    - "keyword_optimization": Keyword optimization score
    
    Also include:
    - "ats_requirements": List of 5 key ATS requirements this resume should meet
    - "missing_elements": List any missing standard resume sections
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
        ats_data = eval(ats_response.choices[0].message.content)
        
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
    - "grammar_score": Grammar accuracy score (0-100)
    - "language_issues": List of specific grammar/language issues found
    - "improvement_suggestions": List of suggestions to improve language
    """
    
    try:
        grammar_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": grammar_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        grammar_data = eval(grammar_response.choices[0].message.content)
        
        scores.update({
            'grammar_score': float(grammar_data.get('grammar_score', 0)),
            'language_issues': grammar_data.get('language_issues', []),
            'improvement_suggestions': grammar_data.get('improvement_suggestions', [])
        })
    except Exception as e:
        st.error(f"Grammar check failed: {str(e)}")
    
    return scores

def visualize_scores(scores: Dict[str, float]):
    """Create visualizations for the scores"""
    st.subheader("📊 Resume Scores")
    
    # Main Score Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ATS Compliance Score", f"{scores['ats_compliance']}/100", 
                 help="How well your resume is formatted for Applicant Tracking Systems")
    with col2:
        st.metric("Content Quality", f"{scores['content_quality']}/100",
                 help="Quality of your resume content and structure")
    with col3:
        st.metric("Grammar Score", f"{scores['grammar_score']}/100",
                 help="Grammar and language accuracy score")
    
    # Keyword Optimization (only if job description provided)
    if 'keyword_optimization' in scores:
        st.metric("Keyword Optimization", f"{scores['keyword_optimization']}/100",
                help="How well your resume matches keywords from the job description")
    
    # Score Breakdown
    with st.expander("Detailed Score Breakdown"):
        st.write("""
        **Score Interpretation:**
        - 90-100: Excellent - Very strong resume
        - 75-89: Good - Some minor improvements needed
        - 60-74: Fair - Several areas need improvement
        - Below 60: Needs significant work
        """)
        
        # Convert scores to dataframe for visualization
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
        
        # Create a clean HTML version of the report for better formatting
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                h2 {{ color: #2C3E50; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                .section {{ margin-bottom: 20px; }}
                .label {{ font-weight: bold; }}
                .score {{ font-weight: bold; color: #3498DB; }}
            </style>
        </head>
        <body>
            <h1>Resume Analysis Report</h1>
            
            <div class="section">
                <h2>Personal Information</h2>
                <p><span class="label">Name:</span> {body.split('Name: ')[1].split('\n')[0] if 'Name: ' in body else 'N/A'}</p>
                <p><span class="label">Email:</span> {body.split('Email: ')[1].split('\n')[0] if 'Email: ' in body else 'N/A'}</p>
                <p><span class="label">Phone:</span> {body.split('Phone: ')[1].split('\n')[0] if 'Phone: ' in body else 'N/A'}</p>
                <p><span class="label">Location:</span> {body.split('Location: ')[1].split('\n')[0] if 'Location: ' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>Resume Scores</h2>
                <p><span class="label">ATS Compliance:</span> <span class="score">{body.split('ATS Compliance: ')[1].split('\n')[0] if 'ATS Compliance: ' in body else 'N/A'}</span></p>
                <p><span class="label">Content Quality:</span> <span class="score">{body.split('Content Quality: ')[1].split('\n')[0] if 'Content Quality: ' in body else 'N/A'}</span></p>
                <p><span class="label">Grammar Score:</span> <span class="score">{body.split('Grammar Score: ')[1].split('\n')[0] if 'Grammar Score: ' in body else 'N/A'}</span></p>
                <p><span class="label">Keyword Optimization:</span> <span class="score">{body.split('Keyword Optimization: ')[1].split('\n')[0] if 'Keyword Optimization: ' in body else 'N/A'}</span></p>
            </div>
            
            <div class="section">
                <h2>Expected Salary Range</h2>
                <p>{body.split('EXPECTED SALARY RANGE')[1].split('RESUME SCORES')[0].strip() if 'EXPECTED SALARY RANGE' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>Professional Timeline</h2>
                <p>{body.split('PROFESSIONAL TIMELINE')[1].split('JOB RECOMMENDATIONS')[0].replace('\n', '<br>').strip() if 'PROFESSIONAL TIMELINE' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>Job Recommendations</h2>
                <p>{body.split('JOB RECOMMENDATIONS')[1].split('SKILLS ANALYSIS')[0].replace('\n', '<br>').strip() if 'JOB RECOMMENDATIONS' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>Skills Analysis</h2>
                <p>{body.split('SKILLS ANALYSIS')[1].split('IMPROVEMENT SUGGESTIONS')[0].replace('\n', '<br>').strip() if 'SKILLS ANALYSIS' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>Improvement Suggestions</h2>
                <p>{body.split('IMPROVEMENT SUGGESTIONS')[1].split('ATS REQUIREMENTS')[0].replace('\n', '<br>').strip() if 'IMPROVEMENT SUGGESTIONS' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>ATS Requirements</h2>
                <p>{body.split('ATS REQUIREMENTS')[1].split('GRAMMAR ISSUES')[0].replace('\n', '<br>').strip() if 'ATS REQUIREMENTS' in body else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>Grammar Issues</h2>
                <p>{body.split('GRAMMAR ISSUES')[1].split('JOB DESCRIPTION MATCHING')[0].replace('\n', '<br>').strip() if 'GRAMMAR ISSUES' in body and 'JOB DESCRIPTION MATCHING' in body else body.split('GRAMMAR ISSUES')[1].replace('\n', '<br>').strip() if 'GRAMMAR ISSUES' in body else 'N/A'}</p>
            </div>
            
            {f'<div class="section"><h2>Job Description Matching</h2><p>{body.split("JOB DESCRIPTION MATCHING")[1].split("CHAT HISTORY")[0].replace(chr(10), "<br>").strip() if "JOB DESCRIPTION MATCHING" in body and "CHAT HISTORY" in body else body.split("JOB DESCRIPTION MATCHING")[1].replace(chr(10), "<br>").strip() if "JOB DESCRIPTION MATCHING" in body else "N/A"}</p></div>' if 'JOB DESCRIPTION MATCHING' in body else ''}
            
            {f'<div class="section"><h2>Chat History with AI Coach</h2><p>{body.split("CHAT HISTORY WITH AI COACH")[1].replace(chr(10), "<br>").strip()}</p></div>' if 'CHAT HISTORY WITH AI COACH' in body else ''}
            
            <p>Generated by Smart Resume Analyzer</p>
        </body>
        </html>
        """
        
        # Attach plain text and HTML versions
        msg.attach(MIMEText(body, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        with smtplib.SMTP_SSL(
            st.secrets["EMAIL_SERVER"], 
            465, 
            timeout=10
        ) as server:
            server.login(
                st.secrets["EMAIL_SENDER"],
                st.secrets["EMAIL_PASSWORD"]
            )
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {str(e)}")
        return False

def create_pdf_report(report_content: str) -> bytes:
    """Generate professional PDF report from text content"""
    class PDF(FPDF):
        def header(self):
            # Add logo if needed
            # self.image('logo.png', 10, 8, 33)
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
    
    # Process report content into structured sections
    sections = {
        'PERSONAL INFORMATION': '',
        'EXPECTED SALARY RANGE': '',
        'RESUME SCORES': '',
        'PROFESSIONAL TIMELINE': '',
        'JOB RECOMMENDATIONS': '',
        'SKILLS ANALYSIS': '',
        'IMPROVEMENT SUGGESTIONS': '',
        'ATS REQUIREMENTS': '',
        'GRAMMAR ISSUES': '',
        'JOB DESCRIPTION MATCHING': '',
        'CHAT HISTORY WITH AI COACH': ''
    }
    
    current_section = None
    for line in report_content.split('\n'):
        if line.strip() in sections:
            current_section = line.strip()
        elif line.strip() and current_section:
            sections[current_section] += line + '\n'
    
    # Add content to PDF with proper formatting
    for section, content in sections.items():
        if not content.strip():
            continue
            
        # Section headings
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, section, 1, 1, 'L', True)
        pdf.ln(2)
        
        # Section content
        pdf.set_font('Arial', '', 11)
        
        # Special formatting for certain sections
        if section == 'PERSONAL INFORMATION':
            for info_line in content.strip().split('\n'):
                if ': ' in info_line:
                    label, value = info_line.split(': ', 1)
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(40, 8, label + ':', 0, 0)
                    pdf.set_font('Arial', '', 11)
                    pdf.cell(0, 8, value, 0, 1)
        elif section == 'RESUME SCORES':
            for score_line in content.strip().split('\n'):
                if ': ' in score_line:
                    label, value = score_line.split(': ', 1)
                    pdf.set_font('Arial', 'B', 11)
                    pdf.cell(50, 8, label + ':', 0, 0)
                    pdf.set_font('Arial', '', 11)
                    pdf.cell(0, 8, value, 0, 1)
        else:
            # Handle multi-line text with proper wrapping
            pdf.multi_cell(0, 6, content.strip())
        
        pdf.ln(5)
    
    # Add generated timestamp
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
    
    try:
        return pdf.output(dest='S').encode('latin1')
    except UnicodeEncodeError:  
        # Handle encoding issues with non-Latin characters
        clean_content = ''.join(c if ord(c) < 256 else ' ' for c in report_content)
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 10, clean_content)
        return pdf.output(dest='S').encode('latin1')

def generate_report_content(personal_info, scores, analysis, salary, job_desc=None, chat_history=None):
    """Create clean, well-formatted report content"""
    report_content = f"""PERSONAL INFORMATION
--------------------
Name: {personal_info.get('name', 'N/A')}
Email: {personal_info.get('email', 'N/A')}
Phone: {personal_info.get('phone', 'N/A')}
Location: {personal_info.get('location', 'N/A')}

EXPECTED SALARY RANGE
--------------------
{salary}

RESUME SCORES
------------
ATS Compliance: {scores.get('ats_compliance', 0)}/100
Content Quality: {scores.get('content_quality', 0)}/100
Grammar Score: {scores.get('grammar_score', 0)}/100
Keyword Optimization: {scores.get('keyword_optimization', 0)}/100

PROFESSIONAL TIMELINE
--------------------
{analysis['timeline']}

JOB RECOMMENDATIONS
------------------
{analysis['recommendations']}

SKILLS ANALYSIS
--------------
{analysis['skills']}

IMPROVEMENT SUGGESTIONS
----------------------
{analysis['feedback']}

ATS REQUIREMENTS
---------------
{chr(10).join(analysis['ats_requirements'])}

GRAMMAR ISSUES
-------------
{chr(10).join(analysis.get('language_issues', ['None found']))}
"""
    
    if job_desc:
        report_content += f"""
JOB DESCRIPTION MATCHING
-----------------------
{analysis.get('jd_match', '')}
"""
    
    # Include chat history in report if exists
    if chat_history and len(chat_history) > 1:
        report_content += "\nCHAT HISTORY WITH AI COACH\n---------------------"
        for msg in chat_history[1:]:  # Skip system prompt
            role = "You" if msg["role"] == "user" else "Career Coach"
            report_content += f"\n\n{role}:\n{msg['content']}"
    
    return report_content




# --------------------- Streamlit UI ---------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "txt", "docx"])
    job_desc = st.text_area("Paste Job Description", height=200,
                          placeholder="Optional: Paste job description for matching...")

# Initialize chat session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main Interface
if uploaded_file:
    resume_text = extract_text(uploaded_file)
    
    if not resume_text:
        st.error("Failed to extract text from file")
        st.stop()
    
    # Personal Info Section
    with st.spinner("Extracting personal information..."):
        personal_info = extract_personal_info(resume_text)
    
    st.header("👤 Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Name", value=personal_info.get('name', 'Not found'), disabled=True)
        st.text_input("Email", value=personal_info.get('email', 'Not found'), disabled=True)
    
    with col2:
        st.text_input("Phone", value=personal_info.get('phone', 'Not found'), disabled=True)
        st.text_input("Location", value=personal_info.get('location', 'Not found'), disabled=True)
    
    # Calculate Scores
    with st.spinner("Calculating ATS and quality scores..."):
        scores = calculate_ats_score(resume_text, job_desc)
        visualize_scores(scores)
    
    # Advanced Analysis
    with st.spinner("Analyzing resume content..."):
        analysis = {
            'timeline': extract_timeline(resume_text),
            'recommendations': generate_job_recommendations(resume_text),
            'skills': analyze_with_openai(f"Categorize skills from: {resume_text[:5000]}\nGroup by:\n- Technical Skills\n- Soft Skills\n- Certifications"),
            'feedback': analyze_with_openai(f"Provide 5 specific improvements for: {resume_text[:5000]}\nInclude:\n- Content suggestions\n- Formatting tips\n- Keyword optimization"),
            'ats_requirements': scores.get('ats_requirements', []),
            'missing_elements': scores.get('missing_elements', []),
            'language_issues': scores.get('language_issues', []),
            'grammar_suggestions': scores.get('improvement_suggestions', [])
        }
        
        # Salary Estimation
        salary = estimate_salary(
            job_title=analysis['recommendations'].split('\n')[0].replace('-', '').strip() if analysis['recommendations'] else "Software Engineer",
            location=personal_info.get('location', 'United States'),
            experience=personal_info.get('experience_years', 3)
        )
        
        if job_desc:
            analysis['jd_match'] = analyze_with_openai(
                f"Compare resume to job description:\nRESUME:{resume_text[:5000]}\nJD:{job_desc}\n"
                "Provide:\n- Match score (0-100%)\n- Missing keywords\n- Suggestions to improve match"
            )
    
    # Display Results
    st.header("📊 Analysis Results")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Timeline", "Job Recs", "Skills", "Feedback", 
        "ATS Requirements", "Grammar", "AI Coach"
    ])
    
    with tab1:
        st.markdown(analysis['timeline'])
    
    with tab2:
        st.markdown(analysis['recommendations'])
    
    with tab3:
        st.markdown(analysis['skills'])
    
    with tab4:
        st.markdown(analysis['feedback'])
    
    with tab5:
        st.subheader("ATS Requirements")
        st.write("These are key elements ATS systems look for:")
        for req in analysis['ats_requirements']:
            st.write(f"- {req}")
        
        if analysis['missing_elements']:
            st.warning("Missing Standard Sections:")
            for missing in analysis['missing_elements']:
                st.write(f"- {missing}")
    
    with tab6:
        st.subheader("Grammar and Language Issues")
        if analysis['language_issues']:
            st.warning("Found these language issues:")
            for issue in analysis['language_issues'][:5]:  # Show top 5 issues
                st.write(f"- {issue}")
        else:
            st.success("No significant grammar issues found!")
        
        if analysis['grammar_suggestions']:
            st.info("Language Improvement Suggestions:")
            for suggestion in analysis['grammar_suggestions'][:3]:
                st.write(f"- {suggestion}")
    
    # Interactive AI Coach Tab
    with tab7:
        st.subheader("💬 Interactive Resume Coach")
        st.write("Ask questions about your resume or get career advice:")
        
        # Display chat history
        for i, msg in enumerate(st.session_state.chat_history):
            if i == 0: continue  # Skip system prompt
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # User input
        if prompt := st.chat_input("Ask about your resume..."):
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.spinner("Thinking..."):
                response, updated_history = chat_with_resume(
                    resume_text,
                    st.session_state.chat_history,
                    prompt
                )
                st.session_state.chat_history = updated_history
            
            with st.chat_message("assistant"):
                st.write(response)
    
    # Salary Estimation Section
    with st.expander("💵 Expected Salary Range"):
        st.write(salary)
    
    if job_desc:
        with st.expander("Job Description Matching"):
            st.markdown(analysis.get('jd_match', ''))
    
    # Report Generation Section
    st.header("📊 Get Your Report")
    email = st.text_input("Enter your email (optional)", value=personal_info.get('email', ''))
    
    # Generate report content
    report_content = f"""
    PERSONAL INFORMATION
    --------------------
    Name: {personal_info.get('name', 'N/A')}
    Email: {personal_info.get('email', 'N/A')}
    Phone: {personal_info.get('phone', 'N/A')}
    Location: {personal_info.get('location', 'N/A')}
    
    EXPECTED SALARY RANGE
    --------------------
    {salary}
    
    RESUME SCORES
    ------------
    ATS Compliance: {scores.get('ats_compliance', 0)}/100
    Content Quality: {scores.get('content_quality', 0)}/100
    Grammar Score: {scores.get('grammar_score', 0)}/100
    Keyword Optimization: {scores.get('keyword_optimization', 0)}/100
    
    PROFESSIONAL TIMELINE
    ---------------------
    {analysis['timeline']}
    
    JOB RECOMMENDATIONS
    -------------------
    {analysis['recommendations']}
    
    SKILLS ANALYSIS
    ---------------
    {analysis['skills']}
    
    IMPROVEMENT SUGGESTIONS
    -----------------------
    {analysis['feedback']}
    
    ATS REQUIREMENTS
    ----------------
    {chr(10).join(analysis['ats_requirements'])}
    
    GRAMMAR ISSUES
    --------------
    {chr(10).join(analysis.get('language_issues', ['None found']))}
    """
    
    if job_desc:
        report_content += f"""
        JOB DESCRIPTION MATCHING
        -----------------------
        {analysis.get('jd_match', '')}
        """
    
    # Include chat history in report if exists
    if len(st.session_state.chat_history) > 1:
        report_content += "\n\nCHAT HISTORY WITH AI COACH\n---------------------"
        for msg in st.session_state.chat_history[1:]:  # Skip system prompt
            role = "You" if msg["role"] == "user" else "Career Coach"
            report_content += f"\n\n{role}:\n{msg['content']}"
    
    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Email Report"):
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
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="resume_analysis_report.pdf",
            mime="application/pdf"
        )

else:
    st.markdown("""
    # Smart Resume Analyzer 💼
    Upload your resume to get:
    - Personal information extraction
    - Professional timeline
    - Job recommendations
    - Skills analysis
    - ATS compliance scoring
    - Grammar and language checking
    - Salary estimation
    - Interactive AI career coaching
    - Email/PDF report options
    
    ## How It Works:
    1. Upload your resume (PDF, DOCX, or TXT)
    2. Optionally add a job description
    3. Get instant analysis and scores
    4. Chat with the AI career coach
    5. Download or email your full report
    """)

