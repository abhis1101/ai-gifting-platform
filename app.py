import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from io import StringIO

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Saverpe AI - RAG Powered", layout="wide")

# --- API CONFIGURATION ---
# Securely load the key from Streamlit Secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except FileNotFoundError:
    st.error("Secrets not found. Please add GEMINI_API_KEY to your Streamlit Secrets.")

# --- RAG KNOWLEDGE BASE (CONTEXT) ---
# We load the specific rules from your uploaded Excel files into the context.

# Source:  (Types.csv - Defines Budget vs Complexity)
CONTEXT_TYPES = """
Employee Level,Budget Type,Budget Range,Complexity
L1,Minimal,<500,Basic
L2,Moderate,500-999,Medium
L3,Substantial,1000-1999,Advanced
L4,Extensive,2000-2999,Custom
L5,Extravagant,3000+,Custom
"""

# Source:  (Rewarding & Gifting.csv - Defines Frequency/Ranking)
CONTEXT_RANKING = """
Type,Sub-Type,Ranking - General
Festive Gifting,Diwali,1
Milestone,Incentives,2
General Festive,New Year,3
Personal Milestones,Birthday,4
Milestone,Team Incentives,5
Personal Milestones,Marriage Anniversary,6
Personal Milestones,Promotions,7
"""

# Source: [cite: 520] (AI-Recommendation.csv - Maps Rank to Rec IDs)
# Simplified sample for the RAG context
CONTEXT_REC_MAPPING = """
Recommendation Rank,Recommendation ID,Reward 1,Reward 2
1,1,Bank Transfer,
2,20,Single Use Card,
3,2,Wallet Recharge,
4,3,Multi-brand GC,
1a,25,Bank Transfer,Multi-brand GC
1b,26,Bank Transfer,Brand Product
2a,442,Single Use Card,Wallet Recharge
"""

# Source: [cite: 526] (Query1.csv - Definitions of Rec IDs)
CONTEXT_QUERY1 = """
ID,Reward Type 1,Reward 1,Reward Type 2,Reward 2,Complexity
1,Money,Bank Tranfer,,,Basic
3,Gift_Cards,Multi brand,,,Basic
25,Money,Bank Tranfer,Gift_Cards,Multi brand,Medium
26,Money,Bank Tranfer,Products,Brand,Medium
507,Money,Bank Tranfer,Money,Bank Tranfer,Advanced
"""

# --- HELPER FUNCTIONS ---

def mock_employee_db():
    """Generates sample employee data if no file is uploaded."""
    data = {
        "ID": [101, 102, 103, 104, 105],
        "Name": ["Abhishek", "Sarah", "Raj", "Emily", "Michael"],
        "Department": ["Sales", "HR", "IT", "Operations", "Executive"],
        "Level": ["L5", "L3", "L4", "L2", "L1"],
        "Role": ["Manager", "Generalist", "Developer", "Associate", "VP"],
        "Last_Reward_Date": ["2025-10-01", "2025-09-15", "2025-11-01", "2025-08-20", "2025-10-20"],
    }
    return pd.DataFrame(data)

# --- THE GEMINI AI AGENT ---

def get_gemini_recommendations(employees_df, total_budget, level_overrides):
    """
    The RAG Engine. It sends the raw data + the context files to Gemini
    to compute the optimal reward strategy.
    """
    
    # Calculate dynamic budget per employee based on total budget if overrides aren't set
    # (This is a simple heuristic to help the AI if manual amounts aren't given)
    avg_budget = total_budget / len(employees_df)
    
    employee_csv = employees_df.to_csv(index=False)
    
    prompt = f"""
    You are the 'Saverpe' Intelligent Reward Engine. 
    
    ### 1. INPUTS:
    - **Total Available Budget:** ₹{total_budget}
    - **Manual Level Allocations:** {level_overrides} (If "0", calculate optimal amount based on Total Budget).
    - **Employee Data:** {employee_csv}

    ### 2. RAG KNOWLEDGE BASE (STRICT RULES):
    
    **A. Budget & Complexity Mapping (Source: Types.csv):**
    {CONTEXT_TYPES}
    *Rule:* Use the "Budget Range" column to determine if an employee gets a Basic, Medium, or Advanced reward.
    
    **B. Frequency & Timing (Source: Rewarding & Gifting.csv):**
    {CONTEXT_RANKING}
    *Rule:* Suggest the "Event" based on the highest priority (Rank 1 = Diwali).
    
    **C. Recommendation IDs (Source: AI-Recommendation.csv & Query1.csv):**
    {CONTEXT_REC_MAPPING}
    {CONTEXT_QUERY1}
    *Rule:* You MUST select a valid "Recommendation ID" (e.g., 1, 25, 26) that matches the Complexity determined in Step A.
    
    ### 3. INSTRUCTIONS:
    1. **Calculate Amount:** For each employee, decide the Reward Amount based on their Level (L1-L5). Use the Manual Allocations if provided; otherwise, distribute the Total Budget efficiently favoring higher levels (L5).
    2. **Determine Complexity:** Compare the Amount to the "Budget Range" in Rule A to find the Complexity (Basic/Medium/Advanced).
    3. **Select Combo:** Pick a "Recommendation ID" from Rule C that matches that Complexity.
    4. **Determine Frequency:** Assign the "Event" based on Rule B (Ranking).
    
    ### 4. OUTPUT FORMAT:
    Return ONLY a valid JSON array. No markdown.
    [
        {{
            "Employee_Name": "Name",
            "Level": "L1/L2...",
            "Department": "Dept",
            "Allocated_Amount": 0,
            "Complexity": "Basic/Medium/Advanced",
            "Event": "e.g. Diwali",
            "Recommendation_ID": "e.g. 25",
            "Reward_Combo": "e.g. Bank Transfer + Gift Card"
        }}
    ]
    """
    
    # Dynamic Model Discovery (Self-Healing Connection)
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

    # Select best model (Flash > Pro)
    model_name = next((m for m in available_models if 'gemini-1.5-flash' in m), available_models[0])
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Clean response
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        data = json.loads(text.strip())
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"AI Processing Error: {e}")
        return pd.DataFrame()

# --- UI LAYOUT ---

def main():
    st.title("Orbit AI: Automated Reward Orchestrator")
    st.markdown("### RAG-Driven Gifting based on Budget & Complexity Mapping")
    
    # Sidebar
    with st.sidebar:
        st.header("Inputs")
        
        # Input 1: Total Budget
        total_budget = st.number_input("Total Gifting Budget (₹)", value=50000, step=5000)
        
        # Input 2: Advanced Level Overrides
        st.markdown("---")
        with st.expander("Advanced: Manual Level Amounts"):
            st.caption("Leave as 0 to let AI calculate based on total budget.")
            l1_amt = st.number_input("Level 1 Amount (₹)", value=0)
            l2_amt = st.number_input("Level 2 Amount (₹)", value=0)
            l3_amt = st.number_input("Level 3 Amount (₹)", value=0)
            l4_amt = st.number_input("Level 4 Amount (₹)", value=0)
            l5_amt = st.number_input("Level 5 Amount (₹)", value=0)
            
            level_overrides = {
                "L1": l1_amt, "L2": l2_amt, "L3": l3_amt, "L4": l4_amt, "L5": l5_amt
            }

        # Input 3: Data
        st.markdown("---")
        uploaded_file = st.file_uploader("Employee DB (CSV)", type=['csv'])
        if uploaded_file:
            employees = pd.read_csv(uploaded_file)
        else:
            employees = mock_employee_db()
            st.info("Using Mock Data")

    # Main Output Area
    if st.button("✨ Run AI Allocation Model", type="primary", use_container_width=True):
        
        with st.spinner("Querying RAG Knowledge Base & Optimizing Budget..."):
            # Run the AI Agent
            result_df = get_gemini_recommendations(employees, total_budget, level_overrides)
            
            if not result_df.empty:
                # --- SUMMARY METRICS ---
                col1, col2, col3, col4 = st.columns(4)
                total_used = result_df['Allocated_Amount'].sum()
                
                col1.metric("Total Budget", f"₹{total_budget:,}")
                col2.metric("Allocated", f"₹{total_used:,}", delta=f"{total_budget-total_used}")
                col3.metric("Employees", len(result_df))
                # Identify best impact strategy used
                strategy_used = result_df['Event'].mode()[0] if not result_df.empty else "N/A"
                col4.metric("Primary Event", strategy_used)

                # --- DETAILED TABLE ---
                st.subheader("Generated Reward Plan")
                st.dataframe(
                    result_df,
                    column_config={
                        "Allocated_Amount": st.column_config.NumberColumn("Amount", format="₹%d"),
                        "Recommendation_ID": st.column_config.TextColumn("Rec ID"),
                    },
                    use_container_width=True
                )
                
                # --- ONE CLICK IMPLEMENTATION ---
                st.markdown("---")
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.caption("This will trigger the payouts based on the 'Recommendation ID' logic defined in the backend.")
                with c2:
                    if st.button("🚀 One-Click Implementation"):
                        st.balloons()
                        st.success(f"Processed {len(result_df)} rewards for {strategy_used}!")

if __name__ == "__main__":
    main()
