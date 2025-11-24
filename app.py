import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from io import StringIO

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Saverpe AI - HR Gifting (Gemini Powered)", layout="wide")

# --- API CONFIGURATION ---
# WARNING: It is generally not safe to hardcode API keys in production apps.
# Ideally, use st.secrets or environment variables.
GEMINI_API_KEY = "AIzaSyCnXfU9G_PiNY93TDoH-4RexbcBrhmH_pM"
genai.configure(api_key=GEMINI_API_KEY)

# --- DATA CONTEXT (RAG LAYER) ---
# These represent your specific business rules and psychological framework constraints.

RULES_DATA = """
Level,Budget_Type,Budget_Range,Complexity
L1,Minimal,<500,Basic
L2,Moderate,500-999,Medium
L3,Substantial,1000-1999,Advanced
L4,Extensive,2000-2999,Custom
L5,Extravagant,3000+,Custom
"""

REWARD_CATALOG = """
ID,Reward_Type,Reward_Name,Complexity,Savings_Potential
1,Money,Bank Transfer,Basic,0%
2,Money,Wallet Recharge,Basic,1%
3,Gift_Cards,Multi-brand,Basic,3%
4,Products,Brand Products,Basic,10%
5,Experiences,Dining/Events,Medium,5%
6,Paid_Time_Off,Extra Leave,Medium,0%
7,Prepaid_Cards,Single Use,Basic,2%
"""

PSYCHOLOGY_CONTEXT = """
1. The 7-Day Rule: Employees must receive recognition every 7 days to maintain engagement.
2. Variable Ratio Schedule: Surprise bonuses work better than expected ones (prevents entitlement).
3. Mental Accounting: Non-cash rewards (gifts/experiences) are valued higher than cash.
4. Social Signaling: Experiential rewards create stories and social currency.
5. Hedonic vs Utilitarian: Sales teams prefer Hedonic (luxury) rewards; Ops may prefer Utilitarian.
"""

# --- HELPER FUNCTIONS ---

def load_data_context():
    return {
        "rules": RULES_DATA,
        "catalog": REWARD_CATALOG,
        "psychology": PSYCHOLOGY_CONTEXT
    }

def mock_employee_db():
    data = {
        "ID": [101, 102, 103, 104, 105],
        "Name": ["Abhishek", "Sarah", "Raj", "Emily", "Michael"],
        "Department": ["Sales", "HR", "IT", "Operations", "Executive"],
        "Level": ["L5", "L3", "L4", "L2", "L1"],
        "Last_Reward_Date": ["2025-10-01", "2025-09-15", "2025-11-01", "2025-08-20", "2025-10-20"],
        "Interests": ["Travel", "Reading", "Gaming", "Cooking", "Golf"]
    }
    return pd.DataFrame(data)

# --- THE GEMINI AI AGENT (DYNAMIC DISCOVERY VERSION) ---

def get_gemini_recommendations(employees_df, total_budget, strategy):
    """
    Dynamically finds an available Gemini model and generates recommendations.
    """
    context = load_data_context()
    employee_csv = employees_df.to_csv(index=False)
    
    prompt = f"""
    You are an expert HR AI Agent for 'Saverpe'.
    
    ### CONTEXT:
    1. Rules: {context['rules']}
    2. Catalog: {context['catalog']}
    3. Psychology: {context['psychology']}
    
    ### INPUT:
    - Budget: {total_budget}
    - Strategy: {strategy}
    - Employees:
    {employee_csv}
    
    ### TASK:
    Recommend rewards for each employee.
    - 'High Impact': Prioritize Experiences/Surprise.
    - 'Best Savings': Prioritize Brand Products (high savings).
    - 'Least Complex': Prioritize Bank Transfer.
    
    ### OUTPUT FORMAT:
    Return ONLY a valid JSON array. No markdown, no explanation text.
    [
        {{
            "Employee": "Name",
            "Level": "Lx",
            "Department": "Dept",
            "Recommended_Reward": "Reward Name",
            "Timing": "Timing",
            "Amount_INR": 0,
            "Reasoning": "Short reason"
        }}
    ]
    """
    
    try:
        # 1. DYNAMICALLY FIND AVAILABLE MODELS
        # We ask Google what models are available for this API key
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        if not available_models:
            st.error("CRITICAL ERROR: No Generative AI models are available for this API Key. Please check your Google Cloud/AI Studio plan.")
            return pd.DataFrame()

        # 2. SELECT THE BEST MODEL
        # Preference: 1.5 Flash (Fast) -> 1.5 Pro (Smart) -> Any Gemini -> Any Available
        selected_model_name = None
        
        # Search strategy
        for m in available_models:
            if 'gemini-1.5-flash' in m:
                selected_model_name = m
                break
        
        if not selected_model_name:
            for m in available_models:
                if 'gemini-1.5-pro' in m:
                    selected_model_name = m
                    break
        
        if not selected_model_name:
            for m in available_models:
                if 'gemini' in m:
                    selected_model_name = m
                    break
                    
        # Fallback to the first one found if nothing matched preferences
        if not selected_model_name:
            selected_model_name = available_models[0]

        # 3. RUN GENERATION
        # st.toast(f"Using AI Model: {selected_model_name}") # Uncomment for debugging
        model = genai.GenerativeModel(selected_model_name)
        response = model.generate_content(prompt)
        
        # 4. CLEAN OUTPUT
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        clean_text = text.strip()
        data = json.loads(clean_text)
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"AI Processing Error: {e}")
        # Print available models to helps debug if it fails again
        # st.write("Debug - Available Models found:", available_models) 
        return pd.DataFrame()

# --- UI LAYOUT ---

def main():
    st.title("Orbit AI: Gifting & Rewarding Engine (Powered by Gemini)")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.success(f"✅ Connected to Gemini API")
        
        st.subheader("Budgeting")
        total_budget = st.number_input("Total Budget (INR)", value=50000, step=5000)
        frequency = st.selectbox("Gifting Frequency", ["Monthly", "Quarterly", "Ad-hoc (Surprise)"])
        
        st.subheader("Data Source")
        uploaded_file = st.file_uploader("Upload Employee DB (CSV)", type=['csv'])
        
        if uploaded_file:
            employees = pd.read_csv(uploaded_file)
        else:
            employees = mock_employee_db()
            st.info("Using Mock Data")

    # Main Area
    with st.expander("📊 Step 1: Review Employee Data", expanded=True):
        st.dataframe(employees, use_container_width=True)

    st.divider()
    st.header("🤖 Step 2: Run AI Recommendation Strategy")
    
    col1, col2, col3 = st.columns(3)
    
    strategy = None
    if col1.button("🚀 High Impact (Psychological)", use_container_width=True):
        strategy = "High Impact"
    if col2.button("💰 Best Savings (Financial)", use_container_width=True):
        strategy = "Best Savings"
    if col3.button("⚡ Least Complex (Operational)", use_container_width=True):
        strategy = "Least Complex"

    if strategy:
        with st.spinner(f"Consulting Gemini AI for '{strategy}' strategy..."):
            rec_df = get_gemini_recommendations(employees, total_budget, strategy)
            
            if not rec_df.empty:
                st.session_state['results'] = rec_df
                st.session_state['strategy'] = strategy

    # Display Results
    if 'results' in st.session_state:
        rec_df = st.session_state['results']
        st.subheader(f"📋 AI Recommendations: {st.session_state['strategy']}")
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        total_spend = rec_df["Amount_INR"].sum()
        # Simple logic to estimate savings based on strategy
        savings_rate = 0.12 if st.session_state['strategy'] == "Best Savings" else 0.04
        savings = total_spend * savings_rate
        
        m1.metric("Total Proposed Spend", f"₹{total_spend:,}")
        m2.metric("Estimated Savings", f"₹{int(savings):,}", f"{savings_rate*100}%")
        m3.metric("Employees Covered", len(rec_df))
        
        # Detailed Table
        st.dataframe(
            rec_df,
            column_config={
                "Amount_INR": st.column_config.NumberColumn("Amount (₹)", format="₹%d"),
            },
            use_container_width=True
        )
        
        # Implementation
        st.divider()
        st.header("✅ Step 3: Execute")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.warning("Clicking 'Approve' will generate payment orders via the connected Vendor API.")
        with c2:
            if st.button("Approve & Pay", type="primary", use_container_width=True):
                st.balloons()
                st.success("Orders processed successfully!")

if __name__ == "__main__":

    main()


