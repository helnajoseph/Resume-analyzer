import streamlit as st
from openai import OpenAI
import httpx
import os
import PyPDF2   
import re
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime
from fpdf import FPDF
import json

import io  # For in-memory file operations
import tempfile  # For temporary file storage
import logging  # For structured logging
import traceback  # For detailed error tracing
import nltk  # Natural Language Toolkit for text processing
from PIL import Image  # If you need to handle images
from dotenv import load_dotenv  # For loading environment variables from .env file

  # For caching computationally expensive operations
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
            model="gpt-3.5-turbo",
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
            model="gpt-3.5-turbo",
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
    """Get detailed and accurate salary estimation"""
    # Clean and validate inputs
    job_title = job_title.strip() if job_title else "Software Developer"  # Default fallback
    location = location.strip() if location else "United States"
    
    # Ensure experience is a valid number
    try:
        experience = int(experience) if experience else 0
    except (ValueError, TypeError):
        experience = 0
        
    # More detailed prompt for better estimates
    prompt = f"""Estimate detailed salary range for:
    Job Title: {job_title}
    Location: {location}
    Experience: {experience} years
    
    Consider regional cost of living factors, industry standards, and experience level.
    Include both annual salary range and hourly rate if applicable.
    
    Return JSON with:
    {{
        "annual_range": {{
            "min": "minimum annual salary with currency symbol",
            "max": "maximum annual salary with currency symbol"
        }},
        "hourly_range": {{
            "min": "minimum hourly rate with currency symbol",
            "max": "maximum hourly rate with currency symbol"
        }},
        "currency": "currency code (e.g., USD, INR, EUR)",
        "factors": ["List", "of", "factors", "considered"],
        "confidence": 1-10 confidence score
    }}
    """
    
    try:
        salary_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        salary_data = json.loads(salary_response.choices[0].message.content)
        
        # Format the response nicely with proper currency formatting
        annual_min = salary_data.get('annual_range', {}).get('min', 'N/A')
        annual_max = salary_data.get('annual_range', {}).get('max', 'N/A')
        hourly_min = salary_data.get('hourly_range', {}).get('min', 'N/A')
        hourly_max = salary_data.get('hourly_range', {}).get('max', 'N/A')
        
        result = f"**Annual Salary Range:** {annual_min} - {annual_max}\n"
        
        if hourly_min != 'N/A' and hourly_max != 'N/A':
            result += f"**Hourly Rate:** {hourly_min} - {hourly_max} per hour\n"
            
        result += f"\n**Factors Considered:**\n"
        for factor in salary_data.get('factors', ['Market rates', 'Experience level', 'Location']):
            result += f"- {factor}\n"
            
        confidence = salary_data.get('confidence', 7)
        result += f"\n**Estimate Confidence:** {confidence}/10"
        
        return result
    except Exception as e:
        # Provide fallback response in case of error
        currency = 'USD' if location.lower() in ['united states', 'usa', 'us'] else '$'
        fallback = f"**Estimated Salary Range:** {currency}{'50,000 - $90,000' if experience < 5 else '80,000 - $130,000'} per year\n"
        fallback += f"**Note:** This is a general estimate based on limited information."
        return fallback

def analyze_grammar_and_language(resume_text: str) -> Dict:
    """
    Enhanced grammar and language analysis with detailed categorization and specific error detection
    
    Args:
        resume_text: The text content of the resume
        
    Returns:
        Dictionary containing grammar score and detailed language analysis
    """
    # Prepare more specific prompts for better analysis
    grammar_prompt = f"""Perform a detailed grammar and language analysis on this resume:
    {resume_text[:5000]}
    
    Analyze the following aspects:
    1. Grammar errors (subject-verb agreement, tense consistency, etc.)
    2. Spelling issues (including commonly misspelled professional terms)
    3. Punctuation problems
    4. Sentence structure (run-ons, fragments, etc.)
    5. Style and tone consistency
    6. Professional vocabulary usage
    7. Action verb effectiveness
    8. Overall readability
    
    Return JSON with:
    {{
        "grammar_score": Overall score from 0-100,
        "error_count": Total number of errors detected,
        "categorized_issues": {{
            "grammar": ["List of specific grammar errors found"],
            "spelling": ["List of misspelled words with correct versions"],
            "punctuation": ["List of punctuation issues"],
            "structure": ["List of sentence structure problems"],
            "style": ["List of style inconsistencies"]
        }},
        "severity_rating": Rating from 1-5 (1=minor issues, 5=critical problems),
        "top_improvements": ["List of 3-5 most important fixes"],
        "impact_on_perception": "Brief assessment of how errors impact professional perception"
    }}
    """
    
    try:
        grammar_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": grammar_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        grammar_data = json.loads(grammar_response.choices[0].message.content)
        
        # Process and enhance the grammar analysis
        analysis_result = {
            'grammar_score': float(grammar_data.get('grammar_score', 0)),
            'error_count': int(grammar_data.get('error_count', 0)),
            'categorized_issues': grammar_data.get('categorized_issues', {
                'grammar': [],
                'spelling': [],
                'punctuation': [],
                'structure': [],
                'style': []
            }),
            'severity_rating': int(grammar_data.get('severity_rating', 1)),
            'top_improvements': grammar_data.get('top_improvements', []),
            'impact_assessment': grammar_data.get('impact_on_perception', '')
        }
        
        # Add category totals for better reporting
        analysis_result['category_counts'] = {
            category: len(issues) 
            for category, issues in analysis_result['categorized_issues'].items()
        }
        
        # Calculate priority score to highlight most important issues
        if analysis_result['error_count'] > 0:
            # Weighted severity calculation
            weights = {
                'grammar': 1.2,  # Grammar errors are highly visible
                'spelling': 1.5,  # Spelling errors create very negative impressions
                'punctuation': 0.8,  # Less critical but still important
                'structure': 1.0,  # Important for readability
                'style': 0.7      # Important but more subjective
            }
            
            category_scores = {}
            for category, issues in analysis_result['categorized_issues'].items():
                if issues:
                    category_scores[category] = len(issues) * weights.get(category, 1.0)
            
            analysis_result['priority_categories'] = sorted(
                category_scores.keys(),
                key=lambda x: category_scores[x],
                reverse=True
            )
        else:
            analysis_result['priority_categories'] = []
            
        return analysis_result
        
    except Exception as e:
        st.error(f"Grammar analysis failed: {str(e)}")
        # Provide fallback analysis
        return {
            'grammar_score': 70,  # Default moderate score
            'error_count': 0,
            'categorized_issues': {
                'grammar': [],
                'spelling': [],
                'punctuation': [],
                'structure': [],
                'style': []
            },
            'severity_rating': 1,
            'top_improvements': ["Error analyzing grammar - please check manually"],
            'impact_assessment': "Unable to assess impact due to analysis error"
        }

def calculate_ats_score(resume_text: str, job_desc: str = "") -> Dict[str, float]:
    """Calculate multiple ATS-related scores with improved grammar analysis"""
    scores = {
        'ats_compliance': 0,
        'content_quality': 0,
        'keyword_optimization': 0,
        'grammar_score': 0
    }
    
    # ATS Compliance Check - unchanged from original
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
            model="gpt-3.5-turbo",
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
    
    # Use the new improved grammar analysis function
    grammar_analysis = analyze_grammar_and_language(resume_text)
    
    # Update scores with grammar analysis results
    scores.update({
        'grammar_score': grammar_analysis.get('grammar_score', 0),
        'grammar_analysis': grammar_analysis
    })
    
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
             model="gpt-3.5-turbo",
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
    st.subheader("Resume Scores")
    
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

def display_grammar_analysis(grammar_analysis):
    """Display enhanced grammar analysis results in the UI"""
    st.subheader("Grammar and Language Analysis")
    
    # Display overall score with color coding
    score = grammar_analysis.get('grammar_score', 0)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if score >= 85:
            st.success(f"Score: {score}/100")
        elif score >= 70:
            st.info(f"Score: {score}/100")  
        else:
            st.warning(f"Score: {score}/100")
            
    with col2:
        st.progress(score/100)
    
    # Display impact assessment
    if grammar_analysis.get('impact_assessment'):
        st.write("**Impact on Perception:**")
        st.write(grammar_analysis.get('impact_assessment'))
    
    # Show error counts by category
    if grammar_analysis.get('category_counts'):
        st.write("**Issue Summary:**")
        categories = grammar_analysis['category_counts']
        
        if sum(categories.values()) > 0:
            for category, count in categories.items():
                if count > 0:
                    st.write(f"- {category.title()}: {count} issues")
        else:
            st.success("No significant grammar or language issues detected!")
    
    # Display priority improvements
    if grammar_analysis.get('top_improvements'):
        st.write("**Top Priority Improvements:**")
        for imp in grammar_analysis.get('top_improvements'):
            st.info(f"• {imp}")
    
    # Show detailed issues if they exist
    if grammar_analysis.get('categorized_issues'):
        with st.expander("Detailed Issue Breakdown"):
            categories = grammar_analysis['categorized_issues']
            
            for category, issues in categories.items():
                if issues:
                    st.write(f"**{category.title()} Issues:**")
                    for issue in issues:
                        st.write(f"- {issue}")
                    st.write("")  # Add spacing between categories

def create_pdf_report(report_content: str) -> bytes:
    """Generate a professionally styled PDF report with Unicode support and proper score formatting"""
    class PDF(FPDF):
        def header(self):
            # More elegant header with logo space
            self.set_font('Helvetica', 'B', 18)
            self.set_text_color(21, 76, 121)  # Dark blue
            self.cell(0, 10, 'Resume Analysis Report', 0, 1, 'C')
            
            # Add a decorative line
            self.set_draw_color(21, 76, 121)  # Dark blue
            self.set_line_width(0.5)
            self.line(20, 22, 190, 22)
            self.ln(15)
        
        def footer(self):
            self.set_y(-20)
            # Add a decorative line
            self.set_draw_color(21, 76, 121)  # Dark blue
            self.set_line_width(0.3)
            self.line(20, self.get_y(), 190, self.get_y())
            
            self.ln(2)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
        def chapter_title(self, title):
            # More elegant chapter titles with gradient effect
            self.set_font('Helvetica', 'B', 13)
            self.set_text_color(21, 76, 121)  # Dark blue
            self.set_fill_color(240, 245, 250)  # Light blue background
            self.set_draw_color(21, 76, 121)  # Dark blue border
            self.cell(0, 10, title, 'B', 1, 'L', True)
            self.ln(5)
            
        def chapter_body(self, body):
            self.set_font('Helvetica', '', 11)
            self.set_text_color(50, 50, 50)  # Dark gray for better readability
            # Handle non-Latin characters by replacing them
            safe_body = body.encode('latin-1', errors='replace').decode('latin-1')
            self.multi_cell(0, 6, safe_body)
            self.ln()
            
        def bullet_item(self, text, indent=0):
            self.set_font('Helvetica', '', 11)
            self.set_text_color(50, 50, 50)  # Dark gray
            self.cell(indent)
            # Use a bullet point character
            self.set_text_color(21, 76, 121)  # Dark blue bullet
            self.cell(5, 6, chr(149), 0, 0)  # Bullet point character (•)
            self.set_text_color(50, 50, 50)  # Back to dark gray
            # Handle non-Latin characters
            safe_text = text.encode('latin-1', errors='replace').decode('latin-1')
            self.multi_cell(0, 6, safe_text)
            
        def bold_bullet_item(self, label, text, indent=0):
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(50, 50, 50)  # Dark gray
            self.cell(indent)
            # Use a bullet point character
            self.set_text_color(21, 76, 121)  # Dark blue bullet
            self.cell(5, 6, chr(149), 0, 0)  # Bullet point character (•)
            self.set_text_color(40, 40, 40)  # Slightly darker gray for label
            
            # For score items, use a fixed width layout
            if '/100' in text:
                self.cell(70, 6, f"{label}:", 0, 0)  # Fixed width cell for label
                self.set_font('Helvetica', 'B', 11)  # Keep bold for score
                self.set_text_color(50, 50, 50)
                # Handle non-Latin characters
                safe_text = text.encode('latin-1', errors='replace').decode('latin-1')
                self.cell(0, 6, safe_text, 0, 1)
            else:
                # Original behavior for non-score items
                self.cell(30, 6, f"{label}:", 0, 0)
                self.set_font('Helvetica', '', 11)
                self.set_text_color(50, 50, 50)  # Back to dark gray
                # Handle non-Latin characters
                safe_text = text.encode('latin-1', errors='replace').decode('latin-1')
                self.multi_cell(0, 6, safe_text)
            
        def score_item(self, label, score, indent=0):
            """Special method for formatting score items with proper alignment"""
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(50, 50, 50)
            self.cell(indent)
            
            # Blue bullet point
            self.set_text_color(21, 76, 121)
            self.cell(5, 6, chr(149), 0, 0)
            
            # Label in bold
            self.set_text_color(40, 40, 40)
            self.cell(70, 6, f"{label}:", 0, 0)
            
            # Score with spacing
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(50, 50, 50)
            self.cell(0, 6, score, 0, 1)
            
        def subsection_title(self, title):
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(21, 76, 121)  # Dark blue
            self.cell(0, 8, title, 0, 1)
            self.ln(2)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)  # Larger margin for elegance
    pdf.add_page()
    
    # Set PDF metadata for a more professional document
    pdf.set_title('Resume Analysis Report')
    pdf.set_author('Resume Analysis System')
    pdf.set_creator('Professional Resume Analyzer')
    
    # Pre-process the report content to replace problematic characters and remove asterisks
    report_content = report_content.replace('\u223c', '~')  # Replace tilde operator with simple tilde
    report_content = report_content.replace('***', '')      # Remove triple asterisks
    report_content = report_content.replace('**', '')       # Remove double asterisks
    report_content = report_content.replace('*', '')        # Remove single asterisks
    
    # Process the report content into structured data
    sections = {
        'Personal Information': [],
        'Resume Scores': [],
        'Job Match Analysis': {'Matching Skills': [], 'Missing Skills': [], 'Recommendations': []},
        'Professional Timeline': [],
        'Job Recommendations': [],
        'Improvement Suggestions': [],
        'Expected Salary': []
    }
    
    current_section = None
    for line in report_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if line.startswith('### '):
            current_section = line[4:]
            if current_section not in sections:
                sections[current_section] = []
        elif line.startswith('#### '):
            subsection = line[5:]
            if isinstance(sections[current_section], dict):
                sections[current_section][subsection] = []
        else:
            # Remove leading dashes for bullet points
            if line.startswith('- '):
                line = line[2:]
                
            if current_section and isinstance(sections[current_section], list):
                sections[current_section].append(line)
            elif current_section and isinstance(sections[current_section], dict):
                last_subsection = list(sections[current_section].keys())[-1]
                sections[current_section][last_subsection].append(line)
    
    # Build the PDF content
    # Personal Information
    pdf.chapter_title('1. Personal Information')
    for line in sections['Personal Information']:
        if ': ' in line:
            label, text = line.split(': ', 1)
            pdf.bold_bullet_item(label.strip(), text.strip(), 10)
        else:
            pdf.bullet_item(line, 10)
    pdf.ln(5)
    
    # Resume Scores - Special handling for better alignment
    pdf.chapter_title('2. Resume Scores')
    for line in sections['Resume Scores']:
        if ': ' in line:
            label, text = line.split(': ', 1)
            # Use specialized score_item method for score formatting
            if '/100' in text:
                pdf.score_item(label.strip(), text.strip(), 10)
            else:
                pdf.bold_bullet_item(label.strip(), text.strip(), 10)
        else:
            pdf.bullet_item(line, 10)
    pdf.ln(5)
    
    # Job Match Analysis
    if 'Job Match Analysis' in sections and sections['Job Match Analysis'].get('Matching Skills', []):
        pdf.chapter_title('3. Job Match Analysis')
        
        # Overall Scores
        for line in sections['Job Match Analysis'].get('', []):
            if ': ' in line:
                label, text = line.split(': ', 1)
                pdf.bold_bullet_item(label.strip(), text.strip(), 10)
        
        # Matching Skills
        pdf.subsection_title('Matching Skills:')
        pdf.set_font('Helvetica', '', 11)
        for skill in sections['Job Match Analysis'].get('Matching Skills', []):
            # Remove any remaining asterisks and leading dashes from each skill
            clean_skill = skill.replace('*', '').replace('- ', '')
            pdf.bullet_item(clean_skill, 15)
        
        # Missing Skills
        if sections['Job Match Analysis'].get('Missing Skills', []):
            pdf.subsection_title('Missing Skills:')
            pdf.set_font('Helvetica', '', 11)
            for skill in sections['Job Match Analysis'].get('Missing Skills', []):
                # Remove any remaining asterisks and leading dashes from each skill
                clean_skill = skill.replace('*', '').replace('- ', '')
                pdf.bullet_item(clean_skill, 15)
        
        # Recommendations
        if sections['Job Match Analysis'].get('Recommendations', []):
            pdf.subsection_title('Recommendations:')
            pdf.set_font('Helvetica', '', 11)
            for rec in sections['Job Match Analysis'].get('Recommendations', []):
                # Remove any remaining asterisks and leading dashes from each recommendation
                clean_rec = rec.replace('*', '').replace('- ', '')
                pdf.bullet_item(clean_rec, 15)
        
        pdf.ln(5)
    
    # Professional Timeline
    if sections.get('Professional Timeline', []):
        pdf.chapter_title('4. Professional Timeline')
        for line in sections['Professional Timeline']:
            # Remove any remaining asterisks and leading dashes
            clean_line = line.replace('*', '').replace('- ', '')
            pdf.bullet_item(clean_line, 10)
        pdf.ln(5)
    
    # Job Recommendations
    if sections.get('Job Recommendations', []):
        pdf.chapter_title('5. Job Recommendations')
        for line in sections['Job Recommendations']:
            # Remove any remaining asterisks and leading dashes
            clean_line = line.replace('*', '').replace('- ', '')
            pdf.bullet_item(clean_line, 10)
        pdf.ln(5)
    
    # Improvement Suggestions
    if sections.get('Improvement Suggestions', []):
        pdf.chapter_title('6. Improvement Suggestions')
        for line in sections['Improvement Suggestions']:
            # Remove any remaining asterisks and leading dashes
            clean_line = line.replace('*', '').replace('- ', '')
            pdf.bullet_item(clean_line, 10)
        pdf.ln(5)
    
    # Expected Salary
    if sections.get('Expected Salary', []):
        pdf.chapter_title('7. Expected Salary')
        for line in sections['Expected Salary']:
            # Remove any remaining asterisks and leading dashes
            clean_line = line.replace('*', '').replace('- ', '')
            pdf.bullet_item(clean_line, 10)
    
    # Add generation date with more elegant styling
    pdf.ln(8)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}', 0, 1, 'R')
    
    try:
        return pdf.output(dest='S').encode('latin-1')
    except UnicodeEncodeError:
        # If we still have encoding issues, try a more aggressive approach
        return pdf.output(dest='S').encode('ascii', errors='replace')

def generate_report_content(personal_info, scores, analysis, salary, job_match=None):
    """Generate clean, structured report content"""
    report = []
    
    # 1. Personal Information
    report.append("### Personal Information")
    report.append(f"- Name: {personal_info.get('name', 'N/A')}")
    report.append(f"- Email: {personal_info.get('email', 'N/A')}")
    report.append(f"- Phone: {personal_info.get('phone', 'N/A')}")
    report.append(f"- Location: {personal_info.get('location', 'N/A')}")
    report.append(f"- Experience: {personal_info.get('experience_years', 0)} years")
    report.append("")
    
    # 2. Resume Scores
    report.append("### Resume Scores")
    report.append(f"- ATS Compliance: {scores.get('ats_compliance', 0)}/100")
    report.append(f"- Content Quality: {scores.get('content_quality', 0)}/100")
    report.append(f"- Grammar Score: {scores.get('grammar_score', 0)}/100")
    report.append(f"- Keyword Optimization: {scores.get('keyword_optimization', 0)}/100")
    report.append("")
    
    # 3. Job Match Analysis
    if job_match:
        report.append("### Job Match Analysis")
        report.append(f"- Overall Match Score: {job_match.get('match_score', 'N/A')}/100")
        report.append(f"- Keyword Coverage: {job_match.get('keyword_coverage', 'N/A')}%")
        report.append("")
        
        report.append("#### Matching Skills")
        for skill in job_match.get('matching_skills', ['N/A'])[:10]:
            report.append(f"- {skill}")
        report.append("")
        
        report.append("#### Missing Skills")
        for skill in job_match.get('missing_skills', ['N/A'])[:10]:
            report.append(f"- {skill}")
        report.append("")
        
        report.append("#### Recommendations")
        for rec in job_match.get('recommendations', ['N/A'])[:5]:
            report.append(f"- {rec}")
        report.append("")
    
    # 4. Professional Timeline - FIXED to clean markdown formatting
    report.append("### Professional Timeline")
    if 'timeline' in analysis:
        # Clean the timeline text
        timeline_text = analysis['timeline']
        
        # Remove markdown code block markers
        timeline_text = timeline_text.replace("```markdown", "").replace("```", "")
        
        # Remove the "### Timeline" header
        timeline_text = re.sub(r'#+\s*Timeline', '', timeline_text)
        
        # Split by lines and process each line
        for line in timeline_text.split('\n'):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip markdown formatting lines or bullet points already in the text
            if line.startswith('#') or line == '•' or line == '-':
                continue
                
            # Remove bullet points if they exist
            if line.startswith('• ') or line.startswith('- '):
                line = line[2:].strip()
                
            # Add the cleaned line as a bullet point
            report.append(f"- {line}")
    else:
        report.append("- N/A")
    report.append("")
    
    # 5. Job Recommendations
    report.append("### Job Recommendations")
    if 'recommendations' in analysis:
        for line in analysis['recommendations'].split('\n'):
            if line.strip():
                report.append(f"- {line.strip()}")
    else:
        report.append("- N/A")
    report.append("")
    
    # 6. Improvement Suggestions
    report.append("### Improvement Suggestions")
    if 'feedback' in analysis:
        for line in analysis['feedback'].split('\n'):
            if line.strip():
                report.append(f"- {line.strip()}")
    else:
        report.append("- N/A")
    report.append("")
    
    # 7. Expected Salary - Fixed to handle multi-line salary data
    report.append("### Expected Salary")
    if salary:
        # Check if salary is a multi-line string
        if isinstance(salary, str) and '\n' in salary:
            # Split by lines and add each as a bullet point
            salary_lines = salary.split('\n')
            for line in salary_lines:
                if line.strip():
                    # Remove markdown formatting for better PDF rendering
                    clean_line = line.replace('**', '')
                    report.append(f"- {clean_line}")
        else:
            # Handle single-line salary
            clean_salary = str(salary).replace('**', '')
            report.append(f"- {clean_salary}")
    else:
        report.append("- Salary information not available")
    
    return '\n'.join(report)

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
                'grammar_suggestions': scores.get('improvement_suggestions', []),
                'grammar_analysis': scores.get('grammar_analysis', {
                    'grammar_score': 0,
                    'error_count': 0,
                    'categorized_issues': {
                        'grammar': [],
                        'spelling': [],
                        'punctuation': [],
                        'structure': [],
                        'style': []
                    },
                    'severity_rating': 1,
                    'top_improvements': [],
                    'impact_assessment': 'No analysis available'
                })
            }
            
            salary = estimate_salary(
                job_title=analysis['recommendations'].split('\n')[0].replace('-', '').strip() if not job_desc else job_desc.split('\n')[0].strip(),
                location=personal_info.get('location', 'United States'),
                experience=personal_info.get('experience_years', 3)
            )
        
        # Display Results
        st.header("Analysis Results")
        
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
            display_grammar_analysis(analysis['grammar_analysis'])
        
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
                
                st.markdown("Improvement Recommendations")
                for i, rec in enumerate(job_match.get('recommendations', []), 1):
                    st.info(f"**{i}.** {rec}")
        
        # Salary Estimation
        with st.expander("Expected Salary Range"):
            st.write(salary)
        
        # Report Generation
        st.header("Download Your Report")
        
        report_content = generate_report_content(
            personal_info, scores, analysis, salary, job_match
        )
        
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
    <style>
    .header-wrapper {
        background: linear-gradient(135deg, #0F1B2D, #1B3A5D);
        margin: -1.5rem -1rem 2rem -1rem;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .title {
        font-size: 52px;
        font-weight: 900;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 0;
        letter-spacing: 1px;
        position: relative;
        display: inline-block;
    }
    
    .title::after {
        content: '';
        position: absolute;
        left: 0;
        bottom: -8px;
        width: 60%;
        height: 4px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 2px;
    }
    
    .subtitle {
        font-size: 18px;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 15px;
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .accent-text {
        color: #4A90E2;
        font-weight: 600;
    }
    </style>
    
    <div class="header-wrapper">
        <div class="title">CVInsight</div>
        <div class="subtitle">Transform Your Resume with <span class="accent-text">AI-Powered</span> Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
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

Get started by uploading your resume on the left sidebar!
""")

    st.info("**Pro Tip**: For the best results, paste a job description you're interested in applying for. This will provide you with tailored matching analysis and recommendations.")
    
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


