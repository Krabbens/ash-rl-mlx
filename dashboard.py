import streamlit as st
import pandas as pd
import json
import altair as alt
import time
import os

st.set_page_config(page_title="RL Loop Dashboard (Ash)", layout="wide")
st.title("🤖 Self-Improving LLM (TerminalBench)")

METRICS_FILE = "metrics.jsonl"

def load_data():
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame(), pd.DataFrame()
    
    tasks = []
    summaries = []
    
    with open(METRICS_FILE, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d["type"] == "task_result":
                    tasks.append(d)
                elif d["type"] == "iteration_summary":
                    summaries.append(d)
            except:
                pass
                
    return pd.DataFrame(tasks), pd.DataFrame(summaries)

placeholder = st.empty()

while True:
    df_tasks, df_summary = load_data()
    
    with placeholder.container():
        if df_tasks.empty:
            st.info("⏳ Waiting for training loop to produce results...")
            time.sleep(2)
            continue
        
        # Calculate live stats from tasks
        current_iter = df_tasks["iteration"].max()
        current_iter_tasks = df_tasks[df_tasks["iteration"] == current_iter]
        live_solved = current_iter_tasks["success"].sum()
        live_total = len(current_iter_tasks)
        live_rate = live_solved / live_total if live_total > 0 else 0
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Iteration", int(current_iter))
        col2.metric("Tasks Processed", f"{live_total}")
        col3.metric("Solved (this iter)", int(live_solved))
        col4.metric("Success Rate", f"{live_rate*100:.1f}%")
        
        # Historical Chart (if we have summaries)
        if not df_summary.empty:
            st.subheader("📈 Performance over Iterations")
            chart = alt.Chart(df_summary).mark_line(point=True).encode(
                x='iteration:O',
                y=alt.Y('success_rate', axis=alt.Axis(format='%'), title='Success Rate'),
                tooltip=['iteration', 'solved_count', 'total_tasks', 'success_rate']
            ).interactive()
            st.altair_chart(chart, width="stretch")
        
        # Live Task View
        st.subheader(f"🔧 Iteration {current_iter} - Task Results")
        
        # Show success/fail table
        display_df = current_iter_tasks[["task_id", "level", "success", "command"]].copy()
        display_df["command"] = display_df["command"].str[:80] + "..."  # Truncate
        st.dataframe(display_df, width="stretch", height=400)
        
        # Success by Category
        st.subheader("📊 Success by Category")
        level_stats = current_iter_tasks.groupby("level")["success"].agg(['sum', 'count']).reset_index()
        level_stats['rate'] = level_stats['sum'] / level_stats['count']
        level_stats.columns = ['Category', 'Solved', 'Total', 'Rate']
        
        level_chart = alt.Chart(level_stats).mark_bar().encode(
            x=alt.X('Category:N', sort='-y'),
            y=alt.Y('Rate:Q', axis=alt.Axis(format='%')),
            color=alt.Color('Rate:Q', scale=alt.Scale(scheme='greens')),
            tooltip=['Category', 'Solved', 'Total', 'Rate']
        )
        st.altair_chart(level_chart, width="stretch")

    time.sleep(2)
