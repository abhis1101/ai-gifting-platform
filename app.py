import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
from io import StringIO

# --- CONFIGURATION ---
st.set_page_config(page_title="Saverpe AI - Intelligent Rewarding", layout="wide")

# --- SECRET MANAGEMENT ---
# Falls back to text input if secrets are not set
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    api_key_configured = True
else:
    api_key_configured = False

# --- RAG KNOWLEDGE BASE (EMBEDDED) ---
# 1. COMPLEXITY & BUDGET MAPPING [Source: Types.csv]
CONTEXT_TYPES = """
Level,Budget_Type,Budget_Range,Complexity
L1,Minimal,<500,Basic
L2,Moderate,500-999,Medium
L3,Substantial,1000-1999,Advanced
L4,Extensive,2000-2999,Custom
L5,Extravagant,3000+,Custom
"""

# 2. EVENT RANKING & FREQUENCY [Source: Rewarding & Gifting.csv]
CONTEXT_RANKING = """
Event,Type,Ranking,Class
Diwali,Festive,1,Normal
Incentives,Milestone,2,Normal
New Year,General Festive,3,Normal
Birthday,Personal,4,Normal
Team Incentives,Milestone,5,Normal
Marriage Anniversary,Personal,6,Normal
Promotions,Personal,7,Normal
"""

# 3. REWARD COMBOS (QUERY1) [Source: Query1.csv]
# A subset of the 10,000+ rows for the AI to choose from
CONTEXT_COMBOS = """
ID,Reward_1,Reward_2,Reward_3,Complexity
1,Bank Transfer,None,None,Basic
2,Wallet Recharge,None,None,Basic
3,Multi-brand GC,None,None,Basic
25,Bank Transfer,Multi-brand GC,None,Medium
42,Bank Transfer,Single use Card,None,Medium
442,Single use Card,Wallet Recharge,None,Medium
554,Bank Transfer,Multi-brand GC,Brand Product,Advanced
593,Bank Transfer,Brand Product,Extra Paid Off,Advanced
507,Bank Transfer,Bank Transfer,Bank Transfer,Advanced
"""

# --- HELPER FUNCTIONS ---

def get_complexity_from_budget(amount):
    # Logic derived from Types.csv 
    if amount < 500: return "Basic"
    if amount < 1000: return "Medium"
    if amount < 3000: return "Advanced"
    return "Custom"

def mock_employee_db():
    # Simulating connected DB
    data = {
        "Employee_ID": ["E001", "E002", "E003", "E004", "E005"],
        "Name": ["Amit", "Sarah", "Raj", "Emily", "Vikram"],
        "Department": ["Sales", "HR", "IT", "Operations", "Management"],
        "Level": ["L5", "L3", "L4", "L2", "L1"], # L1 is Minimal in your sheet, L5 Extravagant
        "Title": ["Sales Manager", "Recruiter", "DevOps Lead", "Logistics Coord", "Intern"],
        "Anniversary": ["2025-10-01", "2025-09-15", "2025-11-01", "2025-08-20", "2025-12-01"]
    }
    return pd.DataFrame(data)

# --- AI AGENT ---

def run_rag_agent(employees, budget_map, strategy):
    """
    Uses Gemini to generate specific reward combos and breakdown.
    """
    
    # Prepare Data Context for the AI
    employee_list_str = employees.to_csv(index=False)
    
    prompt = f"""
    Act as the 'Saverpe' RAG Engine. Your goal is to generate a granular reward plan.
    
    ### KNOWLEDGE BASE (RAG):
    1. **Complexity Rules:** {CONTEXT_TYPES}
    2. **Event Rankings:** {CONTEXT_RANKING}
    3. **Approved Combos:** {CONTEXT_COMBOS}
    
    ### INPUT PARAMETERS:
    - **Budget Allocation per Level:** {json.dumps(budget_map)}
    - **Optimization Strategy:** {strategy}
    - **Employee Data:** {employee_list_str}
    
    ### INSTRUCTIONS:
    For each employee:
    1. **Determine Budget:** Look up their Level (L1-L5) in the Budget Allocation map.
    2. **Determine Complexity:** Use the 'Complexity Rules' based on that budget amount.
    3. **Select Reward Combo:** Pick a 'Recommendation ID' from 'Approved Combos' that matches the calculated Complexity.
    4. **Determine Frequency:** - If Budget is High (>2000): Suggest multiple events (e.g., Diwali + Birthday).
       - If Budget is Low: Suggest top ranked event only (Diwali).
    5. **Calculate Breakup:** specific amounts for each reward in the combo (e.g. if budget 3000, Reward1=2000, Reward2=1000).
    
    ### OUTPUT FORMAT (Valid JSON Array Only):
    [
      {{
        "Name": "Employee Name",
        "Level": "L5",
        "Total_Budget": 3000,
        "Complexity": "Custom",
        "Frequency_Events": "Diwali, Birthday",
        "Rec_ID": 554,
        "Reward_Combo_Name": "Bank Transfer + GC + Product",
        "Amount_Breakup": "Transfer: 1500, GC: 1000, Product: 500"
      }}
    ]
    """
    
    try:
        # Dynamic Model Discovery (Self-Healing)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = 'models/gemini-1.5-flash' if 'models/gemini-1.5-flash' in available_models else available_models[0]
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Cleaning JSON
        text = response.text
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        elif "```" in text: text = text.split("```")[1].split("```")[0]
        
        return pd.DataFrame(json.loads(text.strip()))
        
    except Exception as e:
        st.error(f"AI Error: {e}")
        return pd.DataFrame()

# --- UI MAIN ---

def main():
    # Sidebar - Inputs
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=50)
        st.title("Saverpe Admin")
        
        if not api_key_configured:
            user_key = st.text_input("Enter Gemini API Key", type="password")
            if user_key: genai.configure(api_key=user_key)
        
        st.divider()
        
        # Input Mode Switch
        input_mode = st.radio("Budget Input Mode", ["Simple (Total)", "Advanced (Level-wise)"])
        
        budget_map = {}
        
        if input_mode == "Simple (Total)":
            total_budget = st.number_input("Total Gifting Budget (₹)", value=50000, step=1000)
            # Auto-distribute logic based on  rules
            # Assuming 5 employees for mock; usually you'd divide by headcount
            base = total_budget / 5 
            # Simple weighting: L5 gets most, L1 least (per your CSV logic)
            budget_map = {
                "L1": int(base * 0.5), 
                "L2": int(base * 0.8), 
                "L3": int(base * 1.0), 
                "L4": int(base * 1.2), 
                "L5": int(base * 1.5)
            }
            st.info("💡 Budget auto-distributed based on 'Types.csv' weightings.")
            
        else: # Advanced
            st.subheader("Per Employee Allocation")
            c1, c2 = st.columns(2)
            budget_map["L1"] = c1.number_input("L1 (Minimal)", value=400)
            budget_map["L2"] = c2.number_input("L2 (Moderate)", value=800)
            budget_map["L3"] = c1.number_input("L3 (Substantial)", value=1500)
            budget_map["L4"] = c2.number_input("L4 (Extensive)", value=2500)
            budget_map["L5"] = st.number_input("L5 (Extravagant)", value=3500)

    # Main Area
    st.markdown("## 🎁 Gifting & Rewarding Recommendation Engine")
    
    # 1. Data Load
    employees = mock_employee_db()
    
    # 2. Processing Logic
    if st.button("✨ Generate Recommendations", type="primary"):
        with st.spinner("Consulting RAG Model & Calculating Combos..."):
            
            # We run 3 parallel strategies as requested
            strategies = ["Best Impact", "Best Savings", "Least Complex"]
            tabs = st.tabs(strategies)
            
            for i, strategy in enumerate(strategies):
                with tabs[i]:
                    result_df = run_rag_agent(employees, budget_map, strategy)
                    
                    if not result_df.empty:
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        total_val = result_df['Total_Budget'].sum()
                        m1.metric("Total Cost", f"₹{total_val:,}")
                        m2.metric("Avg Complexity", result_df['Complexity'].mode()[0])
                        m3.metric("Strategy", strategy)
                        
                        # Display The Main Table
                        st.dataframe(
                            result_df,
                            column_config={
                                "Rec_ID": st.column_config.NumberColumn("ID", help="From AI-Recommendation.csv"),
                                "Amount_Breakup": st.column_config.TextColumn("💰 Amount Breakup", width="medium"),
                                "Frequency_Events": st.column_config.TextColumn("📅 Frequency", width="medium"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # One-Click Implementation
                        if st.button(f"🚀 Execute '{strategy}' Plan", key=f"btn_{i}"):
                            st.toast(f"Processing {len(result_df)} rewards...", icon="💳")
                            st.success("Batch payment file generated and sent to Payout Gateway.")
                            st.json(result_df[['Name', 'Total_Budget', 'Rec_ID']].to_dict(orient='records'), expanded=False)

if __name__ == "__main__":
    main()
