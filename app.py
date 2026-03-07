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

tabs = st.sidebar.radio("Go to", ["Introduction", "Proposal Overview", "PDF Overview", "Data overview", "Cleaned data", "Data Exploration", "Inspection and reflection", "Analysis", "Team"])
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

elif tabs == "Analysis":
    st.title("Data analysis")
    st.info("The data analysis will be visible here once completed.")

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

elif tabs == "Data overview":
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

elif tabs == "Cleaned data":
    st.title("Cleaned data")

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

elif tabs == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("Browse a set of curated graphs from the combined cleaned dataset. "
                "Data source: cleaned_data/combined_data_by_year.csv (GitHub).")

    owner = "Soorej30"
    repo = "wage_analysis"
    branch = "main"
    raw_combined = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/cleaned_data/combined_data_by_year.csv"

    @st.cache_data(show_spinner=False)
    def load_combined_github(url):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        # basic cleaning / typing
        # drop aggregated row label
        if "OCC_TITLE" in df.columns:
            df = df[df["OCC_TITLE"].astype(str).str.strip().str.lower() != "all occupations"]
        # coerce numeric fields
        for col in ["TOT_EMP", "H_MEAN", "A_MEAN", "H_MEDIAN",
                    "H_PCT10", "H_PCT25", "H_PCT75", "H_PCT90",
                    "A_PCT10", "A_PCT25", "A_PCT75", "A_PCT90",
                    "EMP_PRSE", "MEAN_PRSE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        # ensure Year is numeric if present
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors='coerce').astype('Int64')
        elif "year" in df.columns:
            df["Year"] = pd.to_numeric(df["year"], errors='coerce').astype('Int64')
        return df

    try:
        combined = load_combined_github(raw_combined)
        # ensure there is a consistent lowercase 'year' column (Int64) for downstream code
        if "Year" in combined.columns:
            combined["year"] = pd.to_numeric(combined["Year"], errors="coerce").astype("Int64")
        elif "year" in combined.columns:
            combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
        else:
            # create empty nullable Int64 column to avoid KeyError
            combined["year"] = pd.Series([pd.NA] * len(combined), dtype="Int64")
    except Exception as e:
        st.error(f"Could not load combined data from GitHub: {e}")
        st.stop()

    st.sidebar.markdown("Visualization controls")
    years_available = sorted([int(y) for y in combined["year"].dropna().unique().tolist()]) if not combined["year"].dropna().empty else []
    year_choice = st.sidebar.selectbox("Select year (or choose All)", ["All"] + [str(int(y)) for y in years_available], index=0)
    top_n = st.sidebar.slider("Top N occupations / states to show", min_value=5, max_value=50, value=20, step=5)

    if year_choice != "All":
        df_v = combined[combined["year"] == int(year_choice)].copy()
    else:
        df_v = combined.copy()
    

    st.markdown("Field descriptions (reference)")
    st.markdown("""
    - AREA: The MSA code or the State fips code (Categorical)  
    - ST / STATE: State abbreviation / name (Categorical)  
    - OCC_CODE / OCC_TITLE: SOC code and title (Categorical)  
    - GROUP: 'major' indicator for major group occupations (Categorical)  
    - TOT_EMP: Estimated total employment (Numerical)  
    - EMP_PRSE / MEAN_PRSE: Relative standard errors (Numerical)  
    - H_MEAN / H_MEDIAN / H_PCT10..90: Hourly stats (Numerical)  
    - A_MEAN / A_MEDIAN / A_PCT10..90: Annual stats (Numerical)  
    - ANNUAL / HOURLY: Flags when only annual/hourly wage is released (Boolean)  
    - Year: Year (Numerical)
    """)

    # 1) Trend: total employment by Year (line)
    if "Year" in combined.columns and "TOT_EMP" in combined.columns:
        agg_year = combined.groupby("Year", dropna=True)["TOT_EMP"].sum().reset_index()
        st.markdown("### Total employment trend by Year")
        st.write("Line chart showing the sum of TOT_EMP per year (all occupations & states).")
        fig1 = px.line(agg_year, x="Year", y="TOT_EMP", markers=True, title="Total TOT_EMP by Year")
        fig1.update_layout(height=500, yaxis_title="Total Employment")
        st.plotly_chart(fig1, use_container_width=True)

    # 2) Top occupations by total employment (bar)
    if "OCC_TITLE" in df_v.columns and "TOT_EMP" in df_v.columns:
        st.markdown("### Top occupations by total employment")
        st.write(f"Bar chart of top {top_n} occupations by summed TOT_EMP (filtered year selection).")
        occ_agg = df_v.groupby("OCC_TITLE", dropna=True)["TOT_EMP"].sum().nlargest(top_n).reset_index()
        fig2 = px.bar(occ_agg, x="TOT_EMP", y="OCC_TITLE", orientation="h", title=f"Top {top_n} OCC_TITLE by TOT_EMP")
        fig2.update_layout(height=700, yaxis={'categoryorder':'total ascending'}, margin=dict(l=250))
        st.plotly_chart(fig2, use_container_width=True)

    # 3) Correlation heatmap for numeric features
    st.markdown("### Correlation heatmap (numeric features)")
    st.write("Shows pairwise Pearson correlation between numeric fields (H_MEAN, H_MEDIAN, TOT_EMP, etc.).")
    numeric = df_v.select_dtypes(include=[np.number]).drop(columns=["Year"], errors='ignore')
    if not numeric.empty:
        corr = numeric.corr()
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Numeric feature correlations")
        fig3.update_layout(height=700)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No numeric columns available for correlation heatmap.")

    # 4) State-level total employment (choropleth-like bar)
    if "STATE" in df_v.columns and "TOT_EMP" in df_v.columns:
        st.markdown("### Total employment by State (top states)")
        st.write(f"Bar chart of top {top_n} states by total TOT_EMP.")
        state_agg = df_v.groupby("STATE", dropna=True)["TOT_EMP"].sum().nlargest(top_n).reset_index()
        fig4 = px.bar(state_agg, x="TOT_EMP", y="STATE", orientation="h", title=f"Top {top_n} STATES by TOT_EMP")
        fig4.update_layout(height=600, yaxis={'categoryorder':'total ascending'}, margin=dict(l=200))
        st.plotly_chart(fig4, use_container_width=True)

    # 5) Pie chart: ANNUAL vs HOURLY rows
    if "ANNUAL" in df_v.columns or "HOURLY" in df_v.columns:
        st.markdown("### Distribution: ANNUAL vs HOURLY flagged rows")
        st.write("Pie chart showing counts of rows flagged ANNUAL vs HOURLY (if present).")
        flags = pd.Series(dtype=int)
        if "ANNUAL" in df_v.columns:
            flags = flags.add(df_v["ANNUAL"].fillna(False).astype(bool).value_counts(), fill_value=0)
        if "HOURLY" in df_v.columns:
            flags = flags.add(df_v["HOURLY"].fillna(False).astype(bool).value_counts(), fill_value=0)
        # fallback: count unique ANNUAL values if booleans present
        if flags.sum() > 0:
            flag_df = flags.rename_axis("flag").reset_index(name="count")
            fig5 = px.pie(flag_df, names="flag", values="count", title="ANNUAL/HOURLY flags distribution")
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No ANNUAL/HOURLY flags detected for pie chart.")

    # 6) Box plot: H_MEDIAN by GROUP
    if "H_MEDIAN" in df_v.columns and "GROUP" in df_v.columns:
        st.markdown("### Hourly median wage by GROUP (box plot)")
        st.write("Box plots of H_MEDIAN for each GROUP (shows spread / outliers).")
        gp = df_v[df_v["GROUP"].notna() & df_v["GROUP"].astype(str).str.strip().ne("")]
        if not gp.empty:
            fig6 = px.box(gp, x="GROUP", y="H_MEDIAN", points="outliers", title="H_MEDIAN distribution by GROUP")
            fig6.update_layout(height=600)
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No GROUP values available to plot H_MEDIAN by group.")

    # 7) Stacked area: employment over time by GROUP (top groups)
    if "Year" in combined.columns and "TOT_EMP" in combined.columns and "GROUP" in combined.columns:
        st.markdown("### Employment over time by GROUP (stacked area)")
        st.write("Area chart of TOT_EMP over years split by top groups (by total employment).")
        grp_tot = combined.groupby("GROUP", dropna=True)["TOT_EMP"].sum().nlargest(6)
        top_groups = grp_tot.index.dropna().tolist()
        if top_groups:
            area_df = (combined[combined["GROUP"].isin(top_groups)]
                       .groupby(["Year", "GROUP"], dropna=True)["TOT_EMP"].sum()
                       .reset_index())
            fig7 = px.area(area_df, x="Year", y="TOT_EMP", color="GROUP", title="TOT_EMP over time by GROUP (top groups)")
            fig7.update_layout(height=700)
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("No GROUP values to build stacked area chart.")

    # 8) Treemap: OCC_TITLE nested by STATE sized by TOT_EMP
    if "OCC_TITLE" in df_v.columns and "STATE" in df_v.columns and "TOT_EMP" in df_v.columns:
        st.markdown("### Treemap: Occupation -> State by TOT_EMP")
        st.write("Treemap showing share of TOT_EMP by occupation and state (top occupations).")
        treemap_df = df_v.groupby(["OCC_TITLE", "STATE"], dropna=True)["TOT_EMP"].sum().reset_index()
        top_occ = treemap_df.groupby("OCC_TITLE")["TOT_EMP"].sum().nlargest(30).index.tolist()
        treemap_df = treemap_df[treemap_df["OCC_TITLE"].isin(top_occ)]
        fig8 = px.treemap(treemap_df, path=["OCC_TITLE", "STATE"], values="TOT_EMP", title="Treemap: OCC_TITLE -> STATE (by TOT_EMP)")
        fig8.update_layout(height=800)
        st.plotly_chart(fig8, use_container_width=True)

    # 9) Scatter (bubble): H_MEDIAN vs TOT_EMP colored by Year, size TOT_EMP
    if "H_MEDIAN" in df_v.columns and "TOT_EMP" in df_v.columns:
        st.markdown("### Scatter: H_MEDIAN vs TOT_EMP (bubble sized by TOT_EMP, color by Year)")
        st.write("Each point is an occupation (or occupation-state) showing relationship between median hourly wage and employment.")
        scatter_df = df_v.groupby(["OCC_TITLE", "Year"], dropna=True).agg(TOT_EMP=("TOT_EMP", "sum"), H_MEDIAN=("H_MEDIAN", "median")).reset_index()
        scatter_df = scatter_df.dropna(subset=["TOT_EMP", "H_MEDIAN"]).nlargest(200, "TOT_EMP")
        fig9 = px.scatter(scatter_df, x="H_MEDIAN", y="TOT_EMP", size="TOT_EMP", color="Year",
                          hover_name="OCC_TITLE", title="H_MEDIAN vs TOT_EMP (top records)")
        fig9.update_layout(height=700)
        st.plotly_chart(fig9, use_container_width=True)

    # 10) Violin / distribution of H_MEDIAN for top 8 states
    if "H_MEDIAN" in df_v.columns and "STATE" in df_v.columns:
        st.markdown("### H_MEDIAN distribution by STATE (violin, top states)")
        state_tot = df_v.groupby("STATE", dropna=True)["TOT_EMP"].sum().nlargest(8).index.tolist()
        viol_df = df_v[df_v["STATE"].isin(state_tot) & df_v["H_MEDIAN"].notna()]
        if not viol_df.empty:
            fig10 = px.violin(viol_df, x="STATE", y="H_MEDIAN", box=True, points="outliers", title="H_MEDIAN distribution for top 8 states")
            fig10.update_layout(height=700)
            st.plotly_chart(fig10, use_container_width=True)
        else:
            st.info("Not enough H_MEDIAN data by state for violin plot.")

    st.markdown("### Notes")
    st.write("Charts limit items (top N) to keep visuals readable. Inspect raw data for full detail. "
             "All numeric columns were coerced to numeric where possible; check for parsing issues if a field is empty.")