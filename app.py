import streamlit as st
from pathlib import Path
import base64
import requests
import urllib.parse
import pandas as pd
import re
import io

# Page Configuration
st.set_page_config(page_title="Group 7 | Wage Variation Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Go to", ["Introduction", "Proposal Overview", "PDF Overview", "Data overview", "Analysis", "Team"])
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
        {"name": "Nivid Pathak", "role": "Data Lead", "bio": "Specializes in data cleaning and ETL processes.", "linkedin": "www.linkedin.com/in/nivid-pathak", "github": "https://github.com/NividPathak", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/aa3deedbafcedc549c97d4bfc18ff36b7840f2f2/images/Nivid_img.jpeg"},
        {"name": "Karan Cheemalapati", "role": "Project Manager", "bio": "Ensures milestone alignment and documentation quality.", "linkedin": "www.linkedin.com/in/karan-cheemalapati", "github" : "https://github.com/karan-cheemalapati", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/8dcf8a44558f4b27531cf5288c6a4c42fc79d3ea/images/Karan_img.JPG"}
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
