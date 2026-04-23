import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="Cereal DEA App", layout="centered")

st.title("🥣 Cereal Efficiency Analyzer (DEA)")
st.write("Select SIP (Small is Preferred → minimize) and LIP (Large is Preferred → maximize)")

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_excel("cereals.xlsx")
    df.columns = df.columns.str.strip()  # fix hidden spaces
    st.success("✅ Data loaded successfully")
except:
    st.error("❌ cereals.xlsx not found")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

columns = df.columns.tolist()

# =========================
# USER INPUT
# =========================
name_col = st.selectbox("Select Cereal Name Column", columns)

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

sip = st.multiselect("Select SIP (Small is Preferred → minimize)", numeric_cols)
lip = st.multiselect("Select LIP (Large is Preferred → maximize)", numeric_cols)

# =========================
# RUN DEA
# =========================
if st.button("Run DEA"):

    if not sip or not lip:
        st.warning("Please select both SIP and LIP variables")
        st.stop()

    # =========================
    # DATA CLEANING
    # =========================
    df_clean = df.copy()

    df_clean = df_clean.replace(["NA", "N/A", ""], None)

    for col in sip + lip:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna(subset=sip + lip)

    st.write(f"Rows used for DEA: {len(df_clean)}")

    if len(df_clean) == 0:
        st.error("❌ No valid data after cleaning")
        st.stop()

    # =========================
    # DEA MODEL
    # =========================
    results = []

    for i in df_clean.index:
        prob = pulp.LpProblem("DEA", pulp.LpMaximize)

        u = pulp.LpVariable.dicts("u", lip, lowBound=0)
        v = pulp.LpVariable.dicts("v", sip, lowBound=0)

        prob += pulp.lpSum([u[o] * df_clean.loc[i, o] for o in lip])

        prob += pulp.lpSum([v[j] * max(df_clean.loc[i, j], 1e-6) for j in sip]) == 1

        for k in df_clean.index:
            prob += (
                pulp.lpSum([u[o] * df_clean.loc[k, o] for o in lip]) -
                pulp.lpSum([v[j] * max(df_clean.loc[k, j], 1e-6) for j in sip])
            ) <= 0

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[status] != "Optimal":
            score = 0
        else:
            score = pulp.value(prob.objective) or 0

        results.append({
            "Cereal": df_clean.loc[i, name_col],
            "Efficiency": round(float(score), 4)
        })

    # =========================
    # RESULTS
    # =========================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Efficiency", ascending=False)

    max_score = results_df["Efficiency"].max()
    if max_score > 0:
        results_df["Efficiency"] = results_df["Efficiency"] / max_score

    st.subheader("🏆 Top 5 Efficient Cereals")
    st.dataframe(results_df.head(5))

    st.subheader("📊 Full Ranking")
    st.dataframe(results_df)

    # =========================
    # FRONTIER GRAPHS (MULTIPLE SIPs)
    # =========================
    st.subheader("📈 Efficiency Frontier Graphs")

    efficient = results_df[results_df["Efficiency"] >= 0.95]
    eff_df = df_clean[df_clean[name_col].isin(efficient["Cereal"])]

    for x in sip:
        st.markdown(f"### {x} vs {lip[0]}")

        fig, ax = plt.subplots()

        # All points
        ax.scatter(df_clean[x], df_clean[lip[0]], label="All Cereals")

        # Efficient points
        ax.scatter(eff_df[x], eff_df[lip[0]], label="Efficient")

        # Frontier curve
        eff_sorted = eff_df.sort_values(by=x)
        ax.plot(eff_sorted[x], eff_sorted[lip[0]])

        # Labels
        for i in eff_sorted.index:
            ax.text(
                eff_sorted.loc[i, x],
                eff_sorted.loc[i, lip[0]],
                str(eff_sorted.loc[i, name_col])[:8],
                fontsize=7
            )

        ax.set_xlabel(x)
        ax.set_ylabel(lip[0])
        ax.set_title("DEA Frontier")
        ax.legend()

        st.pyplot(fig)