"""
Enhanced Streamlit Frontend for Fraud Call Detection
"""

import streamlit as st
import requests
import time
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Fraud Call Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Alert popup */
    .alert-popup {
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 9999;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: slideIn 0.5s ease-out;
        max-width: 400px;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Risk banners */
    .fraud-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(255, 65, 108, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .fraud-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(56, 239, 125, 0.4);
    }
    
    .fraud-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.4);
    }
    
    /* Cards */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Text areas */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'show_alert' not in st.session_state:
    st.session_state.show_alert = False
if 'alert_data' not in st.session_state:
    st.session_state.alert_data = {}


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def show_alert_popup(risk_level, fraud_prob):
    """Show alert popup notification"""
    if risk_level in ["HIGH", "CRITICAL"]:
        alert_class = "alert-high"
        icon = "🚨"
        message = "HIGH RISK DETECTED!"
    elif risk_level == "MEDIUM":
        alert_class = "alert-medium"
        icon = "⚠️"
        message = "MEDIUM RISK"
    else:
        alert_class = "alert-low"
        icon = "✅"
        message = "LOW RISK"
    
    st.markdown(f"""
        <div class="alert-popup {alert_class}">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{message}</div>
            <div style="font-size: 1rem; margin-top: 0.5rem;">
                Fraud Probability: {fraud_prob*100:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)


def display_analysis_results(result):
    """Display analysis results in a formatted way"""
    
    analysis = result.get('analysis', {})
    transcription = result.get('transcription', 'No transcription available')
    
    # Fraud score
    fraud_prob = analysis.get('fraud_probability', 0)
    risk_level = analysis.get('risk_level', 'UNKNOWN')
    
    # Show alert popup
    show_alert_popup(risk_level, fraud_prob)
    
    # Risk level display with color coding
    if risk_level == "CRITICAL" or risk_level == "HIGH":
        st.markdown(f'<div class="fraud-high">🚨 {risk_level} RISK: {fraud_prob*100:.1f}% Fraud Probability</div>', 
                   unsafe_allow_html=True)
    elif risk_level == "MEDIUM":
        st.markdown(f'<div class="fraud-medium">⚡ MEDIUM RISK: {fraud_prob*100:.1f}% Fraud Probability</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fraud-low">✅ LOW RISK: {fraud_prob*100:.1f}% Fraud Probability</div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Transcription - FIXED: removed key parameter
        st.subheader("📝 Transcription")
        st.text_area("Call Content", transcription, height=200, disabled=True)
        
    with col2:
        # Detected Patterns
        st.subheader("🔍 Detected Patterns")
        patterns = analysis.get('detected_patterns', {})
        
        pattern_icons = {
            'urgency_language': ('⏰', 'Urgency Tactics'),
            'financial_requests': ('💳', 'Financial Requests'),
            'impersonation': ('👤', 'Impersonation'),
            'threats': ('⚖️', 'Threats/Intimidation'),
            'verification_request': ('🔐', 'Identity Verification')
        }
        
        for key, (icon, label) in pattern_icons.items():
            if patterns.get(key, False):
                st.error(f"{icon} {label} **DETECTED**")
            else:
                st.success(f"{icon} {label} Not detected")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Security Recommendations")
    recommendations = analysis.get('recommendations', [])
    
    for i, rec in enumerate(recommendations):
        if '🚨' in rec or '⚠️' in rec:
            st.error(rec)
        elif '✅' in rec:
            st.success(rec)
        else:
            st.warning(rec)
    
    # Detailed metrics
    with st.expander("📊 Detailed Analysis Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")
        
        with col2:
            st.metric("Legitimate Probability", f"{analysis.get('legitimate_probability', 0)*100:.2f}%")
        
        with col3:
            patterns_detected = sum(1 for v in patterns.values() if v)
            st.metric("Patterns Detected", f"{patterns_detected}/5")
        
        # AI Reasoning
        if analysis.get('reasoning'):
            st.markdown("**🤖 AI Analysis:**")
            st.info(analysis.get('reasoning'))
            
        # Key Phrases
        if analysis.get('key_phrases'):
            st.markdown("**🔑 Suspicious Phrases:**")
            st.warning(", ".join(analysis.get('key_phrases', [])))

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=100)
    st.title("🛡️ Fraud Detector")
    st.markdown("---")
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info("Start API: `python run_api.py`")
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Analyzed", len(st.session_state.analysis_history))
    fraud_count = sum(1 for item in st.session_state.analysis_history 
                     if item['result']['analysis']['fraud_probability'] > 0.5)
    st.metric("Fraud Detected", fraud_count, delta_color="inverse")
    
    st.markdown("---")
    st.caption("v1.0.0 | Powered by Gemini AI")

# Main Content
st.title("🛡️ Real-Time Fraud Call Detection System")
st.markdown("### Protect yourself from phone scams with AI-powered analysis")

# Upload section
st.markdown("---")
st.subheader("📁 Upload Call Recording for Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "ogg"],
        help="Supported formats: WAV, MP3, M4A, OGG"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # File info
        file_size = len(uploaded_file.getvalue()) / 1024  # KB
        st.caption(f"📄 File: {uploaded_file.name} | Size: {file_size:.1f} KB")

with col2:
    st.info("💡 **Tips:**\n\n"
            "• Clear audio quality\n"
            "• Minimum 5 seconds\n"
            "• English language\n"
            "• Any format accepted")

if uploaded_file is not None:
    analyze_btn = st.button("🔍 Analyze Recording", type="primary", use_container_width=True)
    
    if analyze_btn:
        if not api_status:
            st.error("❌ API is not running! Please start it first.")
        else:
            with st.spinner("🔄 Processing... Transcribing and analyzing..."):
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Upload
                    status_text.text("📤 Uploading file...")
                    progress_bar.progress(20)
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    
                    # Step 2: Transcribe
                    status_text.text("🎤 Transcribing audio with AI...")
                    progress_bar.progress(40)
                    
                    response = requests.post(
                        f"{API_URL}/api/analyze/audio",
                        files=files,
                        timeout=300
                    )
                    
                    # Step 3: Analyze
                    status_text.text("🧠 Analyzing for fraud patterns...")
                    progress_bar.progress(70)
                    
                    if response.status_code == 200:
                        result = response.json()
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Save to history
                        st.session_state.analysis_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': uploaded_file.name,
                            'result': result
                        })
                        
                        # Display Results
                        st.markdown("---")
                        st.subheader("🎯 Analysis Results")
                        display_analysis_results(result)
                        
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.code(response.text)
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Try a smaller audio file.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Analysis History
if len(st.session_state.analysis_history) > 0:
    st.markdown("---")
    st.subheader("📊 Recent Analysis History")
    
    for idx, item in enumerate(reversed(st.session_state.analysis_history[-3:])):  # Show last 3
        with st.expander(f"📞 {item['filename']} - {item['timestamp']}", expanded=False):
            display_analysis_results(item['result'])

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🛡️ Fraud Call Detector v2.0")

with col2:
    st.caption("Made with ❤️ using Streamlit & Gemini AI")

with col3:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")