import streamlit as st
from pathlib import Path
import base64
import requests
import urllib.parse
import pandas as pd
import re
import io
import numpy as np
import plotly.express as px
import json

# Page Configuration
st.set_page_config(page_title="Group 7 | Wage Variation Analysis", layout="wide")

MODELS_IMPLEMENTED = [
    {
        "name": "Random Forest Regressor",
        "category": "Regression",
        "target": "A_MEAN",
        "why": [
            "The target is continuous, so regression is the most direct formulation for wage prediction.",
            "Random forests handle non-linear relationships well across many lagged wage and employment features.",
            "The model is robust to mixed encoded categorical and numeric predictors.",
        ],
        "assumptions": [
            "The historical training data is representative of future wage patterns.",
            "Useful signal exists in non-linear interactions across lagged features.",
            "Very deep trees can overfit, so depth and leaf-size controls matter.",
        ],
        "parameters": {
            "candidate_grid": [
                {"n_estimators": 80, "max_depth": 16, "min_samples_leaf": 1},
                {"n_estimators": 120, "max_depth": 20, "min_samples_leaf": 1},
                {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 3},
            ],
        },
        "metrics": [
            {"metric": "RMSE", "value": None, "direction": "Lower is better", "notes": "Root mean squared error for wage prediction."},
            {"metric": "R2", "value": None, "direction": "Higher is better", "notes": "Explained variance for continuous wage prediction."},
        ],
    },
    {
        "name": "Decision Tree Classifier",
        "category": "Classification",
        "target": "A_MEAN quartile band",
        "why": [
            "A classifier is useful when the question is wage-tier prediction instead of exact wage prediction.",
            "Decision trees are easy to explain because they expose explicit feature splits.",
            "Threshold-based splitting fits wage-band prediction naturally.",
        ],
        "assumptions": [
            "Classes can be separated through recursive feature splits.",
            "The quartile-based target transformation keeps wage tiers meaningful.",
            "Unpruned trees can memorize noise, so complexity controls matter.",
        ],
        "parameters": {
            "candidate_grid": [
                {"max_depth": 6, "min_samples_leaf": 25, "criterion": "gini"},
                {"max_depth": 10, "min_samples_leaf": 25, "criterion": "entropy"},
                {"max_depth": None, "min_samples_leaf": 50, "criterion": "entropy"},
            ],
        },
        "metrics": [
            {"metric": "Accuracy", "value": None, "direction": "Higher is better", "notes": "Overall classification correctness."},
            {"metric": "Precision", "value": None, "direction": "Higher is better", "notes": "Weighted precision across wage bands."},
            {"metric": "Recall", "value": None, "direction": "Higher is better", "notes": "Weighted recall across wage bands."},
            {"metric": "F1-score", "value": None, "direction": "Higher is better", "notes": "Weighted harmonic mean of precision and recall."},
            {"metric": "ROC-AUC", "value": None, "direction": "Higher is better", "notes": "Weighted multiclass one-vs-rest ROC-AUC."},
        ],
    },
    {
        "name": "K-Means",
        "category": "Clustering",
        "target": "Unsupervised segmentation with A_MEAN used for interpretation",
        "why": [
            "K-Means is a strong baseline for grouping jobs into broad wage and labor-market profiles.",
            "The notebook already uses numeric lag features that can be standardized effectively.",
            "Cluster labels can become future engineered features for supervised models.",
        ],
        "assumptions": [
            "Clusters are roughly spherical in scaled feature space.",
            "Feature scaling is necessary because K-Means is distance-based.",
            "The number of clusters must be selected in advance.",
        ],
        "parameters": {
            "candidate_grid": [
                {"n_clusters": 3, "n_init": 20, "random_state": 42},
                {"n_clusters": 4, "n_init": 20, "random_state": 42},
                {"n_clusters": 5, "n_init": 20, "random_state": 42},
                {"n_clusters": 6, "n_init": 20, "random_state": 42},
            ],
        },
        "metrics": [
            {"metric": "Silhouette Score", "value": None, "direction": "Higher is better", "notes": "Measures cohesion and separation between clusters."},
            {"metric": "Davies-Bouldin Index", "value": None, "direction": "Lower is better", "notes": "Measures similarity between clusters; lower is cleaner."},
            {"metric": "Average Target Std", "value": None, "direction": "Higher is better", "notes": "Spread of mean target values across clusters for interpretability."},
        ],
    },
    {
        "name": "FP-Growth",
        "category": "Frequent Pattern Mining",
        "target": "High-pay association discovery",
        "why": [
            "FP-Growth reveals recurring combinations of state, occupation group, and lagged pay bands.",
            "It scales better than Apriori for larger transaction-style datasets.",
            "It adds interpretable high-pay rules even though it is not a direct predictor.",
        ],
        "assumptions": [
            "The data can be converted into basket-style boolean items.",
            "Meaningful structure exists in co-occurring categories and discretized numeric bands.",
            "Support and confidence thresholds strongly shape the rules discovered.",
        ],
        "parameters": {
            "candidate_grid": [
                {"min_support": 0.05, "metric": "confidence", "min_threshold": 0.60},
            ],
        },
        "metrics": [
            {"metric": "Support", "value": None, "direction": "Higher is better", "notes": "How frequently a pattern appears in the data."},
            {"metric": "Confidence", "value": None, "direction": "Higher is better", "notes": "Conditional strength of a rule."},
            {"metric": "Lift", "value": None, "direction": "Higher is better", "notes": "Association strength relative to random co-occurrence."},
            {"metric": "Rule Count", "value": None, "direction": "Higher is better", "notes": "Number of discovered high-pay rules."},
        ],
    },
]


def model_metric_frame():
    rows = []
    for model in MODELS_IMPLEMENTED:
        for metric in model["metrics"]:
            rows.append(
                {
                    "Model": model["name"],
                    "Category": model["category"],
                    "Metric": metric["metric"],
                    "Direction": metric["direction"],
                    "Notebook Value": metric["value"],
                    "Notes": metric["notes"],
                }
            )
    return pd.DataFrame(rows)


def model_parameter_frame():
    rows = []
    for model in MODELS_IMPLEMENTED:
        for param_set in model["parameters"]["candidate_grid"]:
            rows.append(
                {
                    "Model": model["name"],
                    "Category": model["category"],
                    "Parameters": json.dumps(param_set),
                }
            )
    return pd.DataFrame(rows)

# Sidebar Navigation
st.sidebar.title("Navigation")
# quick refresh control to clear cached GitHub/local loads and reload the app
if st.sidebar.button("Refresh data / Clear cache"):
    # clear cached functions decorated with @st.cache_data
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # force a rerun so UI reloads (and cached loads will be re-fetched)
    st.experimental_rerun()

tabs = st.sidebar.radio("Go to", ["Introduction", "Proposal Overview", "PDF Overview", "Uncleaned Data overview", "Data Exploration", "Models Implemented", "Inspection and reflection", "Conclusion", "Team"])
page_width = 1200

if tabs == "Introduction":
    st.title("Labor Market Structure & Wage Prediction")
    st.subheader("Exploring the Drivers of Economic Compensation in the US")

    st.markdown("""
    **Research Topic & Significance:** We study the United States labor market to find the real reasons behind wage dispersion across industries. This matters because a paycheck is the main engine for economic mobility and keeping families afloat. We use official government stats to see why the same skills get a different price tag depending on the sector or location. This work helps students and applicants make choices based on hard numbers rather than just stories or guesses. Policy experts can also use this to spot wage stagnation or regions that need more investment. Our point is to make the link between labor and compensation clear for everyone.

    """, width=page_width, )
    
    st.image("https://raw.githubusercontent.com/Soorej30/wage_analysis/aa3deedbafcedc549c97d4bfc18ff36b7840f2f2/images/labor_dynamics.png", 
             caption="Figure 1: Conceptual visualization of wage distribution across the United States.", width=page_width)

    st.markdown("""
    **Stakeholders:** This research affects a wide range of economic actors from individual employees to large multinational corporations. Students and job applicants are the main beneficiaries because they can use the data to calculate the return on investment for their human capital. Companies and HR departments also gain value by benchmarking their compensation structures against national figures to stay viable in talent acquisition. Economists and groups like the Bureau of Labor Statistics use these interpretations on a macro level to validate their reporting and test labor regulations. Career advisors use the findings to match their programs with industries showing rising wage premiums. Basically anyone involved in the labor force feels the effect of the trends we analyze.

    """, width=page_width)

    st.markdown("""
    **Existing Solutions & Gaps:** Current solutions for wage estimation often rely on static tables or simplified calculators provided by sites like Glassdoor or the BLS's own basic search tools. While these provide a baseline, they frequently fail to account for the interplay between multiple variables, such as how geographic location might amplify or diminish the value of a specific industry code. Literature suggests that while many models exist for predicting income, they often ignore the "Wage Suppression Bias" found in niche occupations where small employment groups lead to data gaps. Research by Smith et al. (2023) highlights that industry aggregation often hides role-specific nuances, leading to generalized predictions that don't apply to specialized technical roles. Our project seeks to fill this gap by using clustering techniques to identify sub-group patterns that traditional regression might overlook.
    """, width=page_width)

    st.markdown("""
    **Blueprint for Your Project:** Our team will execute a multi-phase analytical plan starting with rigorous exploratory data analysis (EDA) to handle missing values and wage suppression. We will then move into feature engineering, where we normalize wages by regional cost-of-living to ensure that high-salary states like California or New York are compared fairly against states with lower overhead. The modeling phase will involve training multiple machine learning algorithms, including Random Forests and Gradient Boosting, to predict Mean Annual Wages. We will also implement unsupervised learning (clustering) to see if we can discover "hidden" categories of jobs that share similar wage structures despite being in different industries. Finally, we will develop an interactive dashboard that allows users to input their occupation and location to receive a predicted wage percentile.
    """, width=page_width)

    st.markdown("""
    **Dataset Considerations:** Our analysis runs on the Occupational Employment and Wage Statistics dataset from the Bureau of Labor Statistics. This source is robust and includes around 800 occupations along with variables like the mean and various wage percentiles. We plan to add regional Consumer Price Index data to improve our geographic adjustments. Using public OEWS data keeps things transparent even though we have to be careful with suppressed values in small employment pockets. Merging these rich sources allows us to build a predictive model that is accurate and grounded.
    """, width=page_width)

elif tabs == "Proposal Overview":
    st.title("Project Scope & Research Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Scope Summary")
        st.write("- **Goal:** Predict and explain wage variation using ML.")
        st.write("- **Data:** ~800 occupations from the US BLS.")
        st.write("- **Key Variables:** Industry Code, Location, and Wage Percentiles.")
        st.write("- **Techniques:** Regression, Clustering, and EDA.")

    with col2:
        st.image("https://raw.githubusercontent.com/Soorej30/wage_analysis/aa3deedbafcedc549c97d4bfc18ff36b7840f2f2/images/flowchart.png", 
                 caption="Project Methodology Workflow")

    st.divider()
    
    st.header("Research Questions")
    questions = [
        "1. Does the total number of people working in a field have a correlation with the mean annual wage occupation-wise?",
        "2. If we look at the data right now which five industries are paying the best median wage for people in entry level roles?",
        "3. Do we see a massive swing in pay for the exact same job title just by looking at different state data?",
        "4. Is there a statistically significant wage gap between 'Management' and 'Technical' roles within the same industry? ",
        "5. Can we accurately predict the 90th percentile wage of an occupation based on its 10th percentile and industry code? ",
        "6. Which geographic regions exhibit the highest 'wage-to-cost-of-living' ratio? ",
        "7. Can clustering identify groups of occupations that are underpaid relative to their required education levels?",
        "8. How has the wage gap between the 10th and 90th percentiles changed across the top 10 industries? ",
        "9. To what extent does 'Industry Aggregation' obscure wage differences in specialized sub-roles?  ",
        "10. Can we predict if an occupation belongs to a 'High Growth' category based solely on its current wage percentiles? "
    ]

    for q in questions:
        st.write(q)

elif tabs == "Models Implemented":
    st.title("Models Implemented")
    st.caption("A guided summary of the models built in `wage_analysis/notebooks/models.ipynb`.")

    st.markdown("### Modeling Scope")
    st.write(
        "The notebook uses `A_MEAN` as the primary prediction target because annual mean wage is smoother and easier to interpret than "
        "`H_MEAN`, while staying closely aligned with hourly wage behavior. The five-year lag features make the dataset a strong fit for "
        "supervised prediction, segmentation, and association mining."
    )

    st.markdown("### Model Cards")
    for model in MODELS_IMPLEMENTED:
        with st.expander(f"{model['name']} ({model['category']})", expanded=True):
            col_a, col_b = st.columns([1.35, 1.0])
            with col_a:
                st.markdown(f"**Prediction focus:** {model['target']}")
                st.markdown("**Why this model was chosen**")
                for reason in model["why"]:
                    st.write(f"- {reason}")
                st.markdown("**Core assumptions**")
                for assumption in model["assumptions"]:
                    st.write(f"- {assumption}")

            with col_b:
                st.markdown("**Metrics used in the notebook**")
                metric_df = pd.DataFrame(model["metrics"])[["metric", "direction", "notes"]]
                metric_df.columns = ["Metric", "Direction", "Meaning"]
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

                st.markdown("**Parameters explored**")
                param_df = pd.DataFrame(model["parameters"]["candidate_grid"])
                st.dataframe(param_df, use_container_width=True, hide_index=True)

    st.markdown("### All Metrics from the Notebook")
    metrics_df = model_metric_frame()
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("### Parameter Summary")
    st.dataframe(model_parameter_frame(), use_container_width=True, hide_index=True)

    st.markdown("### Final Comparison Visualization")
    st.write(
        "These models belong to different ML families, so their raw scores are not directly comparable on a single scale. "
        "The visual comparison below focuses on metric coverage and evaluation breadth across models."
    )

    coverage_df = metrics_df.assign(Covered=1).pivot(index="Model", columns="Metric", values="Covered").fillna(0)
    fig_heatmap = px.imshow(
        coverage_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=["#edf4fb", "#1f5f8b"],
        title="Metric Coverage by Model",
    )
    fig_heatmap.update_layout(height=520, margin=dict(l=40, r=40, t=80, b=40), coloraxis_showscale=False)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    metric_count_df = (
        metrics_df.groupby(["Model", "Category"], as_index=False)["Metric"]
        .count()
        .rename(columns={"Metric": "Metric Count"})
    )
    fig_counts = px.bar(
        metric_count_df,
        x="Model",
        y="Metric Count",
        color="Category",
        text="Metric Count",
        title="How Many Metrics Each Model Reports",
    )
    fig_counts.update_layout(height=480, margin=dict(l=40, r=40, t=80, b=80))
    st.plotly_chart(fig_counts, use_container_width=True)

# elif tabs == "Analysis":
#     st.title("Data analysis")
#     st.info("The data analysis will be visible here once completed.")

elif tabs == "Conclusion":
    st.title("Conclusion & Key Findings")
    st.caption("A synthesis of modeling results and visualization findings from notebooks/models.ipynb and notebooks/Visualizations.ipynb.")

    PINK_SCALE = ["#fce4ec", "#f48fb1", "#ec407a", "#ec0a55", "#a3003b"]
    PLOT_BG = "#fff0f5"
    GRID_C = "#f9c6d8"
    FONT_C = "#3d0018"

    def pink_layout(fig, height=440):
        fig.update_layout(
            height=height,
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            font_color=FONT_C,
            margin=dict(l=40, r=40, t=70, b=50),
        )
        fig.update_xaxes(showgrid=True, gridcolor=GRID_C)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_C)
        return fig

    # ── Section 1: Model Conclusions ─────────────────────────────────────────
    st.markdown("## Model Conclusions")
    st.markdown(
        "Four models were trained on 341,705 rows of BLS occupational wage data with five years of "
        "lag features. Each model addressed a distinct question about the labor market."
    )

    # RF + DT side by side
    col1, col2 = st.columns(2)
    with col1:
        rf_params = ["80 est · depth 16 · leaf 1", "120 est · depth 20 · leaf 1", "120 est · depth ∞ · leaf 3"]
        rf_r2 = [0.970035, 0.969420, 0.970788]
        rf_df = pd.DataFrame({"Params": rf_params, "R²": rf_r2})
        rf_fig = px.bar(
            rf_df, x="R²", y="Params", orientation="h",
            text="R²", color="R²",
            color_continuous_scale=PINK_SCALE,
            title="Random Forest : R² by Parameter Set",
        )
        rf_fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        rf_fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        pink_layout(rf_fig, height=360)
        st.plotly_chart(rf_fig, use_container_width=True)
        st.markdown(
            "**Best:** R² = **0.9708**, RMSE = **$5,234** (120 estimators, uncapped depth, leaf≥3). "
            "Five years of historical wage percentiles explain 97% of wage variance. Labor markets "
            "exhibit strong inertia. Past wages are the best predictor of future wages."
        )

    with col2:
        dt_params = ["depth 6 · gini · leaf 25", "depth 10 · entropy · leaf 25", "depth ∞ · entropy · leaf 50"]
        dt_acc = [0.899833, 0.894083, 0.896750]
        dt_df = pd.DataFrame({"Params": dt_params, "Accuracy": dt_acc})
        dt_fig = px.bar(
            dt_df, x="Accuracy", y="Params", orientation="h",
            text="Accuracy", color="Accuracy",
            color_continuous_scale=PINK_SCALE,
            title="Decision Tree : Accuracy by Parameter Set",
        )
        dt_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        dt_fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        pink_layout(dt_fig, height=360)
        st.plotly_chart(dt_fig, use_container_width=True)
        st.markdown(
            "**Best:** Accuracy = **89.98%**, F1 = **0.8999** (depth 6, Gini). A shallow tree "
            "correctly assigns ~90% of occupation-state pairs to the right wage quartile with "
            "full interpretability, the shallowest and simplest model performs best."
        )

    # K-Means + Cluster profile
    col3, col4 = st.columns(2)
    with col3:
        km_df = pd.DataFrame({
            "k": [3, 4, 5, 6],
            "Silhouette": [0.2352, 0.2351, 0.2341, 0.2280],
            "Davies-Bouldin": [1.481, 1.357, 1.299, 1.210],
        })
        km_fig = px.line(
            km_df, x="k", y=["Silhouette", "Davies-Bouldin"],
            markers=True,
            title="K-Means : Cluster Quality by k",
            color_discrete_sequence=["#ec0a55", "#a3003b"],
        )
        km_fig.update_traces(line=dict(width=2.5))
        pink_layout(km_fig, height=360)
        st.plotly_chart(km_fig, use_container_width=True)
        st.markdown(
            "**Best k = 3** (Silhouette = 0.2352). Despite similar silhouette scores across k values, "
            "k=3 yields the cleanest wage separation and highest cluster wage spread. "
            "Adding more clusters fragments the low-wage group without meaningful gain."
        )

    with col4:
        cp_df = pd.DataFrame({
            "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2 (High)"],
            "Mean Annual Wage": [46036, 46916, 109041],
            "Avg Hourly (prev yr, $)": [21.19, 21.62, 49.51],
            "Avg EMP_PRSE": [109.75, 465.18, 275.77],
        })
        cp_fig = px.bar(
            cp_df, x="Cluster", y="Mean Annual Wage",
            text="Mean Annual Wage",
            color="Mean Annual Wage",
            color_continuous_scale=PINK_SCALE,
            title="K-Means : Cluster Wage Profiles (k=3)",
        )
        cp_fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        cp_fig.update_layout(coloraxis_showscale=False)
        pink_layout(cp_fig, height=360)
        st.plotly_chart(cp_fig, use_container_width=True)
        st.markdown(
            "Two low-wage clusters (\~\$46K) are separated mainly by employment precision error "
            "(EMP_PRSE), not wage level. Cluster 2 is a clear high-wage group at **\$109K**, "
            "a \$63K annual gap separates the labor market into two distinct tiers."
        )

    # FP-Growth
    st.markdown("### FP-Growth : High-Pay Association Rules")
    fp_df = pd.DataFrame({
        "Antecedents": ["high_hist_hourly", "detailed group + high_hist_hourly", "high_hist_hourly", "high_hist_hourly + large_emp"],
        "Consequents": ["high_pay", "high_pay", "detailed group + high_pay", "high_pay"],
        "Support": [0.211, 0.199, 0.199, 0.052],
        "Confidence": [0.845, 0.844, 0.797, 0.835],
        "Lift": [3.381, 3.375, 3.350, 3.340],
    })
    st.dataframe(fp_df, use_container_width=True, hide_index=True)
    st.markdown(
        "All 4 rules share a single dominant antecedent: **high historical hourly pay**. "
        "An occupation with high past hourly wages is **3.38× more likely** to be in the top pay quartile "
        "(confidence 84.5%). Employment size adds marginal lift. This confirms that wage mobility is "
        "strongly path-dependent. High-pay occupations stay high-pay."
    )

    st.divider()

    # ── Section 2: Visualization Findings ─────────────────────────────────────
    st.markdown("## Visualization Findings")
    st.caption("Charts recreated from notebooks/Visualizations.ipynb using computed summary statistics.")

    # 2a: Entry-level wages
    st.markdown("### Top Industries by Entry-Level Wage (10th Percentile)")
    entry_df = pd.DataFrame({
        "Occupation Group": [
            "Management Occupations",
            "Architecture and Engineering",
            "Computer and Mathematical",
            "Business and Financial Operations",
            "Legal Occupations",
            "Life, Physical, and Social Science",
            "Healthcare Practitioners and Technical",
            "Arts, Design, Entertainment, Sports",
            "Education, Training, and Library",
            "Community and Social Service",
        ],
        "Median A_PCT10 ($)": [46335, 42650, 40625, 36875, 34990, 33250, 31870, 27440, 25680, 23210],
    })
    entry_fig = px.bar(
        entry_df, x="Median A_PCT10 ($)", y="Occupation Group", orientation="h",
        text="Median A_PCT10 ($)",
        color="Median A_PCT10 ($)",
        color_continuous_scale=PINK_SCALE,
        title="Top 10 Occupation Groups by Entry-Level Annual Wage (A_PCT10 Median, 2009–2023)",
    )
    entry_fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    entry_fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"), margin=dict(r=120))
    pink_layout(entry_fig, height=480)
    st.plotly_chart(entry_fig, use_container_width=True)
    st.markdown(
        "**Management occupations offer the highest entry-level floor at $46,335.** Architecture & Engineering "
        "and Computer & Mathematical follow. Even at the 10th percentile, STEM and management fields "
        "offer substantially better compensation than service-oriented categories at the bottom of this list."
    )

    st.markdown("### Management vs Technical Wage Gap")
    gap_df = pd.DataFrame({
        "Role": ["Management (11-xxxx)", "Technical (15-xxxx, 17-xxxx)"],
        "Median Annual Wage": [91660, 74470],
        "Mean Annual Wage": [94908, 76524],
        "Sample Size (n)": [35423, 49065],
    })
    c_a, c_b = st.columns([2, 1])
    with c_a:
        import plotly.graph_objects as go
        gap_fig = go.Figure()
        gap_fig.add_trace(go.Bar(
            name="Median Wage", x=gap_df["Role"], y=gap_df["Median Annual Wage"],
            marker_color="#ec0a55", text=gap_df["Median Annual Wage"],
            texttemplate="$%{text:,.0f}", textposition="outside",
        ))
        gap_fig.add_trace(go.Bar(
            name="Mean Wage", x=gap_df["Role"], y=gap_df["Mean Annual Wage"],
            marker_color="#a3003b", text=gap_df["Mean Annual Wage"],
            texttemplate="$%{text:,.0f}", textposition="outside",
        ))
        gap_fig.update_layout(
            barmode="group",
            title="Management vs Technical : Median & Mean Annual Wage",
            yaxis_title="Annual Wage (USD)",
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            font_color=FONT_C, height=400,
            margin=dict(l=40, r=40, t=70, b=50),
        )
        gap_fig.update_yaxes(showgrid=True, gridcolor=GRID_C)
        st.plotly_chart(gap_fig, use_container_width=True)
    with c_b:
        st.metric("Management Median", "$91,660")
        st.metric("Technical Median", "$74,470")
        st.metric("Gap", "$17,190", delta="Management leads by 18.7%")
        st.metric("Mann-Whitney p-value", "< 0.0001", delta="Statistically significant ✓")
    st.markdown(
        "Management roles earn **$17,190 more at the median** than Computer & Mathematical / "
        "Architecture & Engineering roles. A Mann-Whitney U test (n=84,488 total rows) confirms "
        "the gap is statistically significant (p ≈ 0). This holds across all years in the dataset."
    )

    st.markdown("### Total Employment vs Mean Annual Wage (Occupation-Level)")
    scatter_data = [
        ("Retail Salespersons", 49440, 28065), ("Cashiers", 43480, 22980),
        ("Fast Food and Counter Workers", 40195, 26040), ("Combined Food Prep & Serving", 37040, 19470),
        ("Registered Nurses", 34820, 71330), ("Customer Service Representatives", 30710, 36125),
        ("General and Operations Managers", 30530, 110670), ("Stockers and Order Fillers", 30075, 33805),
        ("Laborers and Freight", 29750, 30860), ("Office Clerks, General", 29580, 33985),
        ("Waiters and Waitresses", 28755, 23705), ("Home Health and Personal Care Aides", 27380, 29030),
        ("Janitors and Cleaners", 26605, 28060), ("Secretaries and Admin Assistants", 25590, 37305),
        ("Heavy Truck Drivers", 24260, 46015), ("Stock Clerks", 21510, 25815),
        ("Bookkeeping and Accounting Clerks", 19755, 40325), ("Nursing Assistants", 19230, 29700),
        ("First-Line Supervisors Office", 18890, 56585), ("Elementary School Teachers", 16950, 56765),
        ("Teaching Assistants", 16245, 31410), ("Retail Supervisors", 16015, 44705),
        ("Maintenance Workers General", 15375, 41120), ("Sales Reps Wholesale", 15275, 67630),
        ("Cooks Restaurant", 15070, 27115), ("Personal Care Aides", 14860, 22020),
        ("Assemblers", 14425, 38510), ("Accountants and Auditors", 13170, 72345),
        ("Food Prep Supervisors", 13110, 35835), ("Secondary School Teachers", 12580, 57915),
        ("Software Developers", 12455, 116330), ("Receptionists", 12275, 29625),
        ("Light Truck Drivers", 12090, 42795), ("Software Developers & QA", 11915, 99650),
        ("Security Guards", 11830, 31320), ("Construction Laborers", 11665, 38510),
        ("Project Management Specialists", 11665, 76920), ("Team Assemblers", 11550, 30310),
        ("Maids and Housekeeping Cleaners", 11265, 24535), ("Medical Assistants", 8140, 34090),
        ("Police Patrol Officers", 8490, 58650), ("Dental Hygienists", 2770, 74225),
        ("Database Administrators", 1025, 84360), ("Family Practitioners (MD)", 1340, 193020),
        ("Economics Teachers", 180, 108930), ("Operations Research Analysts", 990, 81980),
        ("Materials Scientists", 140, 96945), ("Orthopedic Surgeons", 200, 339365),
        ("Fashion Designers", 140, 66535), ("Emergency Management Directors", 150, 73960),
        ("Coaches and Scouts", 2895, 44590), ("Fitness Trainers", 2950, 35770),
    ]
    scat_df = pd.DataFrame(scatter_data, columns=["OCC_TITLE", "Avg_TOT_EMP", "Avg_A_MEAN"])
    m_s, b_s = -0.946822, 62792.93
    x_line = np.linspace(0, scat_df["Avg_TOT_EMP"].max(), 200)
    y_line = m_s * x_line + b_s

    scat_fig = go.Figure()
    scat_fig.add_trace(go.Scatter(
        x=scat_df["Avg_TOT_EMP"], y=scat_df["Avg_A_MEAN"],
        mode="markers",
        marker=dict(color="#ec0a55", size=7, opacity=0.6, line=dict(color="#7d0030", width=0.5)),
        text=scat_df["OCC_TITLE"],
        hovertemplate="<b>%{text}</b><br>Employment: %{x:,.0f}<br>Wage: $%{y:,.0f}<extra></extra>",
        name="Occupation",
    ))
    scat_fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="#a3003b", width=2, dash="dash"),
        name="OLS trend",
    ))
    scat_fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.97,
        text="Pearson r = −0.107  (p < 0.001)<br>Spearman r = −0.130  (p < 0.001)",
        showarrow=False, align="right",
        bgcolor="#fce4ec", bordercolor="#ec0a55", borderwidth=1,
        font=dict(color=FONT_C, size=12),
    )
    scat_fig.update_layout(
        title="Median Total Employment vs Median Annual Wage : Detailed Occupations",
        xaxis_title="Median Total Employment (across states & years)",
        yaxis_title="Median Annual Mean Wage (USD)",
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font_color=FONT_C, height=520,
        margin=dict(l=60, r=60, t=70, b=50),
    )
    scat_fig.update_xaxes(showgrid=True, gridcolor=GRID_C)
    scat_fig.update_yaxes(showgrid=True, gridcolor=GRID_C)
    st.plotly_chart(scat_fig, use_container_width=True)
    st.markdown(
        "**Pearson r = −0.107, Spearman r = −0.130** (both p < 0.001 across 973 detailed occupations). "
        "The negative correlation confirms that **high-employment occupations tend to pay less**. "
        "Retail Salespersons, Cashiers, and Fast Food Workers anchor the top-right/bottom area. "
        "Surgeons, Software Developers, and Economists cluster in the low-employment, high-wage corner. "
        "Wage is driven by skill scarcity, not labor demand volume."
    )

    st.divider()

    # ── Section 3: Overall Conclusions ───────────────────────────────────────
    st.markdown("## Overall Conclusions")
    st.markdown("""
**From the models (models.ipynb):**

- **Wage history dominates all other features.** The Random Forest's R² of 0.97 is achieved almost entirely through five years of lagged wage percentiles. This means the labor market is strongly mean-reverting. An occupation that paid well last year will almost certainly pay well next year.
- **Wage-tier classification is a solved problem.** A decision tree with depth 6 achieves 90\% accuracy on wage-quartile prediction. This is production-quality performance using only lag features and occupation/state identifiers.
- **Two labor market tiers exist, not a spectrum.** K-Means consistently finds a clean break between a large low-wage tier (\~\$46K) and a much smaller high-wage tier (\~\$109K). The $63K gap between them is not gradual, it is a structural divide.
- **Being in a historically high-paying field is the single best predictor of future high pay.** FP-Growth confirms this with 84.5\% confidence and 3.38× lift. Occupation choice, not individual effort within a field, is the dominant driver of compensation.

**From the visualizations (Visualizations.ipynb):**

- **Management occupations have the highest entry-level floor** ($46,335 at the 10th percentile), followed by Architecture & Engineering and Computer & Mathematical. These are the best fields for minimizing financial risk at career start.
- **Management earns 18.7% more than Technical roles at the median** (\$91,660 vs \$74,470), and this gap is statistically confirmed (Mann-Whitney p ~ 0). Leadership commands a significant premium over technical expertise.
- **Nominal wages are misleading for geographic comparison.** After cost-of-living adjustment, Midwestern states (Illinois, Michigan, Minnesota) offer better purchasing power than coastal states. California's nominal advantage disappears entirely once CoL is factored in.
- **Large workforces signal lower wages.** Across 973 occupations, higher employment correlates negatively with wages (r = −0.107). Fields that employ millions (retail, food service, care work) are systematically the lowest paid, regardless of the work's social value.

**Limitations:**

- CoL index values are approximate (MERIC 2023); US territories were excluded.
- The five-year lag requirement reduces the usable dataset to rows with data from 2009 onward.
- Wage suppression in small employment cells introduces downward bias in some occupation-state combinations.
- No education-level or occupation-growth data was merged, limiting fairness analysis.
""")

elif tabs == "Team":
    st.title("Meet Group 7")
    st.info("**Mission Statement:** To provide clarity in the labor market through transparent, data-driven analysis of wage structures.")
    
    team_members = [
        {"name": "Soorej S Nair", "role": "Modeling Lead", "bio": "Focused on regression analysis and predictive accuracy.", "linkedin": "https://www.linkedin.com/in/soorej-s-nair-73559470/", "github": "https://github.com/Soorej30", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/aa3deedbafcedc549c97d4bfc18ff36b7840f2f2/images/Soorej_img.jpeg"},
        {"name": "Anjana Anand", "role": "Visualization Lead", "bio": "Expert in creating intuitive and interactive dashboards.", "linkedin":"https://www.linkedin.com/in/anjana-anand-b63076398/", "github": "https://github.com/anjana5anand", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/aa3deedbafcedc549c97d4bfc18ff36b7840f2f2/images/Anjana_img.jpeg"},
        {"name": "Nivid Pathak", "role": "Data Lead", "bio": "Specializes in data cleaning and ETL processes.", "linkedin": "https://www.linkedin.com/in/nivid-pathak", "github": "https://github.com/NividPathak", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/aa3deedbafcedc549c97d4bfc18ff36b7840f2f2/images/Nivid_img.jpeg"},
        {"name": "Karan Cheemalapati", "role": "Project Manager", "bio": "Ensures milestone alignment and documentation quality.", "linkedin": "https://www.linkedin.com/in/karan-cheemalapati", "github" : "https://github.com/karan-cheemalapati", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/8dcf8a44558f4b27531cf5288c6a4c42fc79d3ea/images/Karan_img.JPG"}
    ]
    
    for member in team_members:
        col_img, col_txt = st.columns([1, 4])
        with col_img:
            try:
                st.image(member["img_url"], caption=member['name'])
            except:
                st.image("https://via.placeholder.com/150", caption=member['name'])

        with col_txt:
            st.subheader(member['name'])
            st.write(f"**Role:** {member['role']}")
            st.write(member['bio'])
            st.markdown(f"[LinkedIn]({member['linkedin']}) | [GitHub]({member['github']})")
        st.divider()

elif tabs == "PDF Overview":
    st.title("Our project overview")

    pdf_github_raw = "https://raw.githubusercontent.com/Soorej30/wage_analysis/main/files/Milestone%200_%20Project%20Proposal%20Group%207.pdf"

    viewer_url = f"https://docs.google.com/gview?url={urllib.parse.quote_plus(pdf_github_raw)}&embedded=true"

    try:
        st.components.v1.iframe(viewer_url, height=820)
        st.markdown(f"[Open PDF in a new tab]({pdf_github_raw})")
        st.download_button("Download PDF from GitHub", requests.get(pdf_github_raw, timeout=10).content, file_name=Path(pdf_github_raw).name)
    except Exception as e:
        st.warning(f"Could not embed PDF viewer: {e}")
        try:
            resp = requests.get(pdf_github_raw, timeout=10)
            resp.raise_for_status()
            pdf_bytes = resp.content
            st.download_button("Download PDF from GitHub", pdf_bytes, file_name=Path(pdf_github_raw).name)
            st.markdown(f"Open the PDF in a new tab: [Open PDF]({pdf_github_raw})")
        except Exception as e2:
            st.error(f"Could not fetch PDF: {e2}")
            st.info("Update `pdf_github_raw` to a reachable raw URL or place a local PDF in files/ and use the download button.")

    # Local fallback
    local_pdf = Path("files/Milestone 0_ Project Proposal Group 7.pdf")
    if local_pdf.exists():
        pdf_bytes = local_pdf.read_bytes()
        st.download_button("Download packaged PDF", pdf_bytes, file_name=local_pdf.name)

elif tabs == "Uncleaned Data overview":
    st.title("Our data overview")


    owner = "Soorej30"
    repo = "wage_analysis"
    branch = "main"
    base_path = "data"

    @st.cache_data(show_spinner=False)
    def github_list(path):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    try:
        root = github_list(base_path)
    except Exception as e:
        st.error(f"Could not list GitHub data/ folder: {e}")
        st.info("Check repo/branch and network access.")
        root = []

    year_files = {}
    for item in root:
        if item.get("type") != "dir":
            continue
        name = item.get("name", "")
        m = re.search(r'oesm(\d+)st', name, re.IGNORECASE)
        if not m:
            continue
        digits = m.group(1)
        year = int(digits) if len(digits) > 2 else 2000 + int(digits)
        try:
            dir_contents = github_list(f"{base_path}/{name}")
        except Exception:
            dir_contents = []
        files = [
            {"name": f["name"], "path": f["path"]}
            for f in dir_contents
            if f.get("type") == "file"
            and f["name"].lower().startswith("state_")
            and f["name"].lower().endswith((".xls", ".xlsx"))
        ]
        if files:
            year_files[str(year)] = sorted(files, key=lambda x: x["name"])

    if not year_files:
        st.info("No matching `oesm<number>st` subfolders with `state_*.xls*` files found on GitHub under data/.")
    else:
        years = sorted(year_files.keys(), reverse=True)
        selected_year = st.selectbox("Select year (GitHub)", years, index=0)
        files_for_year = year_files[selected_year]

        if len(files_for_year) > 1:
            names = [f["name"] for f in files_for_year]
            chosen_name = st.selectbox("Choose file", names, index=0)
            file_obj = next(f for f in files_for_year if f["name"] == chosen_name)
        else:
            file_obj = files_for_year[0]

        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_obj['path']}"
        st.write(f"Showing GitHub file for year {selected_year} — {file_obj['name']}")
        st.markdown(f"[Open raw file in new tab]({raw_url})")

        @st.cache_data(show_spinner=False)
        def load_excel_github(raw_url):
            r = requests.get(raw_url, timeout=20)
            r.raise_for_status()
            return pd.read_excel(io.BytesIO(r.content))

        try:
            df = load_excel_github(raw_url)
            st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
            st.dataframe(df, use_container_width=True)
            # provide download button for convenience
            r = requests.get(raw_url, timeout=20)
            st.download_button("Download this XLS file", r.content, file_name=file_obj["name"])
        except Exception as e:
            st.error(f"Could not load Excel from GitHub: {e}")
            st.info("You can open the raw file link above.")

elif tabs == "Data Exploration":
    st.title("Data Exploration (Source - https://www.bls.gov/oes/tables.htm )")

    source = st.selectbox("Data source", ["Local", "GitHub"], index=0)
    owner = "Soorej30"
    repo = "wage_analysis"
    branch = "main"
    cleaned_path = "cleaned_data"

    # Local listing
    if source == "Local":
        cleaned_dir = Path("cleaned_data")
        if not cleaned_dir.exists():
            st.warning(f"cleaned_data folder not found: {cleaned_dir.resolve()}")
        else:
            year_files = {}
            for p in sorted(cleaned_dir.iterdir()):
                if p.is_file() and p.name.lower().startswith("data_") and p.suffix.lower() == ".csv":
                    m = re.search(r'data_(\d{4})', p.name, re.IGNORECASE)
                    if m:
                        year = m.group(1)
                        year_files[year] = p

            if not year_files:
                st.info("No data_<year>.csv files found in cleaned_data/.")
            else:
                years = sorted(year_files.keys(), reverse=True)
                selected_year = st.selectbox("Select year", years, index=0)
                chosen_path = year_files[selected_year]
                st.write(f"Showing cleaned data for year {selected_year} — {chosen_path.name}")

                @st.cache_data(show_spinner=False)
                def load_csv_local(path):
                    try:
                        return pd.read_csv(path)
                    except Exception as e:
                        return e

                df_or_err = load_csv_local(chosen_path)
                if isinstance(df_or_err, Exception):
                    st.error(f"Error reading CSV file: {df_or_err}")
                else:
                    df = df_or_err
                    st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
                    st.dataframe(df, use_container_width=True)
                    with open(chosen_path, "rb") as f:
                        st.download_button("Download this CSV file", f.read(), file_name=chosen_path.name)

                # Field descriptions table (show local file if present, otherwise try GitHub raw)
                st.markdown("### Field descriptions")
                field_desc_local = cleaned_dir / "field_descriptions.csv"
                if field_desc_local.exists():
                    try:
                        fdesc = pd.read_csv(field_desc_local)
                        st.dataframe(fdesc, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not read local field_descriptions.csv: {e}")
                else:
                    # fallback to GitHub raw
                    raw_fd = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/cleaned_data/field_descriptions.csv"
                    try:
                        r = requests.get(raw_fd, timeout=10)
                        r.raise_for_status()
                        fdesc = pd.read_csv(io.BytesIO(r.content))
                        st.dataframe(fdesc, use_container_width=True)
                        st.markdown(f"[Open field_descriptions.csv on GitHub]({raw_fd})")
                    except Exception:
                        st.info("field_descriptions.csv not found locally or on GitHub.")

                # -----------------------
                # Summary statistics + similarity / integration
                # -----------------------
                st.markdown("## Summary statistics")
                num = df.select_dtypes(include=[np.number])
                if not num.empty:
                    stats = pd.DataFrame({
                        "mean": num.mean(),
                        "median": num.median(),
                        "variance": num.var(),
                        "std": num.std(),
                        "skewness": num.skew(),
                        "count": num.count()
                    })
                    st.dataframe(stats.round(4), use_container_width=True)
                else:
                    st.info("No numeric columns available for summary statistics.")

                st.markdown("## Data similarity & integration")
                # if there are other available cleaned years, compare to the previous year if present
                try:
                    available_years = sorted(year_files.keys(), reverse=True)
                except Exception:
                    available_years = []
                similarity_msg = "No comparison data available."
                if available_years and selected_year in available_years:
                    idx = available_years.index(selected_year)
                    if idx < len(available_years) - 1:
                        prev_year = available_years[idx + 1]
                        prev_path = year_files[prev_year]
                        prev_df = None
                        try:
                            prev_df = load_csv_local(prev_path)
                        except Exception:
                            prev_df = None
                        if isinstance(prev_df, pd.DataFrame):
                            # compare numeric column means (intersection)
                            n1 = df.select_dtypes(include=[np.number])
                            n2 = prev_df.select_dtypes(include=[np.number])
                            common = n1.columns.intersection(n2.columns)
                            if len(common) >= 1:
                                v1 = n1[common].mean().fillna(0)
                                v2 = n2[common].mean().fillna(0)
                                # Pearson correlation between mean vectors
                                corr = v1.corr(v2)
                                jaccard = len(set(df.columns).intersection(set(prev_df.columns))) / len(set(df.columns).union(set(prev_df.columns)))
                                st.write(f"Compared to previous available year: {prev_year}")
                                st.write(f"- Pearson correlation of numeric-column means: {corr:.4f}")
                                st.write(f"- Column-name Jaccard similarity: {jaccard:.4f}")
                            else:
                                st.info("No common numeric columns to compare with previous year.")
                        else:
                            st.info("Previous year file could not be loaded for comparison.")
                    else:
                        st.info("No earlier cleaned year available for comparison.")
                else:
                    st.info(similarity_msg)

                st.markdown("---")
                st.markdown("## Visualizations")

                # 1) Heatmap of correlations
                try:
                    if not num.empty:
                        corr = num.corr()
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title="Feature Correlation Heatmap"
                        )
                        fig.update_layout(height=800, margin=dict(l=40, r=40, t=80, b=40))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No numeric columns to compute correlations.")
                except Exception as e:
                    st.warning(f"Could not create correlation heatmap: {e}")

                # 2) Pie chart for STATE vs number of rows
                try:
                    if "STATE" in df.columns:
                        state_counts = df["STATE"].fillna("Unknown").value_counts()
                        fig = px.pie(
                            names=state_counts.index,
                            values=state_counts.values,
                            title="Distribution of rows by STATE"
                        )
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                        fig.update_layout(height=700, margin=dict(l=40, r=40, t=80, b=40))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Column 'STATE' not found for pie chart.")
                except Exception as e:
                    st.warning(f"Could not create state pie chart: {e}")

                # 3) Stacked bar chart of OCC_TITLE vs count stacked by STATE (top N OCC_TITLE)
                try:
                    if "OCC_TITLE" in df.columns and "STATE" in df.columns:
                        top_n = 10
                        occ_counts = df.groupby("OCC_TITLE").size().sort_values(ascending=False).head(top_n)
                        top_occs = occ_counts.index.tolist()
                        pivot = (
                            df[df["OCC_TITLE"].isin(top_occs)]
                            .groupby(["OCC_TITLE", "STATE"])
                            .size()
                            .reset_index(name="count")
                        )
                        fig = px.bar(
                            pivot,
                            x="OCC_TITLE",
                            y="count",
                            color="STATE",
                            title=f"Top {top_n} OCC_TITLE counts stacked by STATE"
                        )
                        fig.update_layout(barmode="stack", xaxis={'categoryorder': 'total descending'}, height=800, margin=dict(l=60, r=40, t=80, b=200))
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Columns 'OCC_TITLE' and/or 'STATE' missing for stacked bar.")
                except Exception as e:
                    st.warning(f"Could not create stacked bar chart: {e}")

                # 4) Bubble chart (updated): OCC_TITLE on x, STATE on y, bubble size = TOT_EMP, color = H_MEDIAN
                try:
                    needed = {"OCC_TITLE", "STATE", "TOT_EMP", "H_MEDIAN"}
                    if needed.issubset(set(df.columns)):
                        # aggregate by occupation + state
                        agg = (
                            df.groupby(["OCC_TITLE", "STATE"])
                            .agg(TOT_EMP=("TOT_EMP", "sum"), H_MEDIAN=("H_MEDIAN", "mean"))
                            .reset_index()
                        )

                        # limit to top occupations by total employment to avoid clutter
                        top_occs = agg.groupby("OCC_TITLE")["TOT_EMP"].sum().nlargest(30).index.tolist()
                        agg = agg[agg["OCC_TITLE"].isin(top_occs)]

                        fig = px.scatter(
                            agg,
                            x="OCC_TITLE",
                            y="STATE",
                            size="TOT_EMP",
                            color="H_MEDIAN",
                            hover_name="OCC_TITLE",
                            title="OCC_TITLE vs STATE — bubble size = TOT_EMP, color = H_MEDIAN",
                            size_max=120
                        )
                        fig.update_layout(height=1300, margin=dict(l=60, r=40, t=80, b=200))
                        fig.update_xaxes(tickangle=-45, automargin=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Columns 'OCC_TITLE', 'STATE', 'TOT_EMP', and 'H_MEDIAN' are required for this bubble chart.")
                except Exception as e:
                    st.warning(f"Could not create bubble chart: {e}")
                st.markdown("---")

    st.markdown("## Additional Graphs")
    st.markdown("The following figures are from `images/Graphs` and provide extra perspective on employment and wage patterns.")

    graph_entries = [
        ("Emp_cnt_vs_year.png", "It looks like the number of jobs have steadily increased except for the drop in 2009 (Probably because of the great recession) and the next drop in 2020 (Probably during Covid 19 time)."),
        ("occ_Vs_emp_cnt.png", "Office and Administrative support has the most number of employees, followed by Food preparation and serving related occupations."),
        ("heatmap.png", "The correlation heatmap is as expected. The regions with high correlation is expected since the percentile values are usually correlated."),
        ("state_vs_emp_cnt.png", "California and New York are highest employees, as expected since they have the most jobs."),
        ("wage_box_plot.png", "The Hourly median wage is slightly higher for detailed group compared to major group."),

        ("emp_vs_year.png", "The Total employees in detailed group increased after 2011."),
        ("stateVsEmp_tree.png", "This shows the different occupation counts across different states and different sizes showing different population count. This shows the trend we observed earlier. California usually has the highest population count in most categories."),
        ("wageVsEmp_Scatter.png", "Scatter plot of wage versus employment to inspect relationships between pay levels and job volume. We can see the trend of increasing median hourly salary as the year increases."),
        ("wageVsState_Violin.png", "Violin plot of wage distributions by state, capturing spread and density of pay values.The wage distribution for each state is similar as expected."),
        ("meadian_hourly_QQ.png", "Q-Q plot for median hourly wages to assess how closely wages follow a normal distribution. Both the Q-Q plots show skeweness."),
    ]

    graphs_dir = Path("images/Graphs")
    for image_name, explanation in graph_entries:
        local_image_path = graphs_dir / image_name
        github_image_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/images/Graphs/{image_name}"
        st.image(str(local_image_path) if local_image_path.exists() else github_image_url, caption=image_name, use_container_width=True)
        st.write(explanation)
        st.markdown("")


    st.info("""
        Reflection: The occupational wage dataset appears to have wide variation in column availability across years and many missing values for some categorical indicators (e.g., GROUP, ANNUAL, HOURLY). \n
        Duplicated OCC_CODE rows are present in many files but titles are largely consistent where duplicates occur. \n
        Numeric distributions differ between years for some variables (see numeric_similarity).\n
        Small sample sizes in some cells reduce reliability of distributional comparisons. \n
        Potential biases include - undercoverage of small occupations or geographic areas, reporting artifacts due to changes in survey design or coding across years, and rounding conventions. \n
        These limitations mean downstream analyses (e.g., trend estimation or pay-gap studies) should carefully filter for comparable fields/years, consider sample sizes, and propagate uncertainty. \n
        Sensitive use: avoid overinterpreting small subgroups and be transparent about missingness and cleaning steps.    \n
    """)

elif tabs == "Inspection and reflection":
    st.title("Inspection & Reflection")

    owner = "Soorej30"
    repo = "wage_analysis"
    branch = "main"
    reports_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/cleaned_data/reports"

    @st.cache_data(show_spinner=False)
    def load_json_github(raw_url):
        r = requests.get(raw_url, timeout=15)
        r.raise_for_status()
        return r.json()

    @st.cache_data(show_spinner=False)
    def load_text_github(raw_url):
        r = requests.get(raw_url, timeout=15)
        r.raise_for_status()
        return r.text

    inspection = None
    reflection_text = None

    # Load inspection JSON from GitHub
    try:
        inspection = load_json_github(f"{reports_base}/consolidated_inspection.json")
    except Exception as e:
        st.error(f"Could not load consolidated_inspection.json from GitHub: {e}")
        st.stop()

    # Try JSON reflection first, fallback to TXT
    try:
        reflection_json = load_json_github(f"{reports_base}/consolidated_reflection.json")
        reflection_text = json.dumps(reflection_json, indent=2)
    except Exception:
        try:
            reflection_text = load_text_github(f"{reports_base}/consolidated_reflection.txt")
        except Exception:
            reflection_text = None

    # years available
    years = sorted(inspection.get("per_year", {}).keys(), reverse=True)
    if not years:
        st.info("No per_year data found in consolidated_inspection.json")
        st.stop()

    st.sidebar.markdown("## Inspection controls")
    selected_years = st.sidebar.multiselect("Select 1 or 2 years to inspect / compare", years, default=[years[0]])
    if not selected_years:
        st.info("Choose at least one year from the sidebar.")
        st.stop()

    with st.expander("Raw consolidated_inspection.json (preview)"):
        st.json(inspection)

    if reflection_text:
        with st.expander("Reflection / notes"):
            st.text(reflection_text)

    def numeric_means_for_year(year):
        ny = inspection["per_year"].get(year, {})
        ns = ny.get("numeric_summary", {})
        means = {}
        for col, stats in ns.items():
            m = stats.get("mean")
            if m is not None:
                try:
                    means[col] = float(m)
                except Exception:
                    continue
        return pd.Series(means, name=year)

    if len(selected_years) == 1:
        y = selected_years[0]
        st.subheader(f"Summary for {y}")
        per = inspection["per_year"].get(y, {})
        st.write(f"Rows: {per.get('n_rows')} — Columns: {per.get('n_cols')}")
        st.markdown("### Missingness (top keys)")
        missing = per.get("missing_pct", {})
        if missing:
            miss_df = pd.Series(missing, name="missing_pct").sort_values(ascending=False).head(20)
            st.dataframe(miss_df.to_frame(), use_container_width=True)
        st.markdown("### Numeric means (top 20 by mean)")
        s = numeric_means_for_year(y)
        if not s.empty:
            s_sorted = s.sort_values(ascending=False).head(20).to_frame("mean").round(3)
            st.dataframe(s_sorted, use_container_width=True)
            fig = px.bar(s_sorted.reset_index(), x="index", y="mean", title=f"Top numeric means — {y}")
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric summary for selected year.")

    elif len(selected_years) == 2:
        y1, y2 = selected_years
        st.subheader(f"Comparison: {y1} vs {y2}")
        s1 = numeric_means_for_year(y1)
        s2 = numeric_means_for_year(y2)
        common = s1.index.intersection(s2.index)
        if len(common) == 0:
            st.info("No common numeric columns to compare between the two years.")
        else:
            comp_df = pd.DataFrame({y1: s1[common], y2: s2[common]}).dropna()
            st.markdown("### Means for common numeric columns (sample)")
            st.dataframe(comp_df.round(3).sort_values(by=y1, ascending=False).head(50), use_container_width=True)

            long = comp_df.reset_index().melt(id_vars="index", value_vars=[y1, y2], var_name="year", value_name="mean")
            fig = px.bar(long, x="index", y="mean", color="year", barmode="group", title=f"Mean comparison — {y1} vs {y2}")
            fig.update_layout(height=700, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            try:
                corr = comp_df[y1].corr(comp_df[y2])
                st.write(f"Pearson correlation between mean vectors: {corr:.4f}")
            except Exception:
                st.info("Could not compute correlation between mean vectors.")
