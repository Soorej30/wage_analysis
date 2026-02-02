import streamlit as st

# Page Configuration
st.set_page_config(page_title="Group 7 | Wage Variation Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Go to", ["Introduction", "Proposal Overview", "Team"])

# --- TAB 1: INTRODUCTION ---
if tabs == "Introduction":
    st.title("Labor Market Structure & Wage Prediction")
    st.subheader("Exploring the Drivers of Economic Compensation in the US")

    # Paragraph 1: Research Topic & Significance
    st.markdown("""
    **Research Topic & Significance:** Our research focuses on the complex dynamics of the United States labor market, specifically targeting the structural factors that dictate wage variation across different sectors. 
    Understanding these patterns is critical because wages are the primary driver of economic mobility and household stability for millions of Americans. By leveraging official labor statistics, we aim to decode why 
    similar skill sets might yield vastly different financial outcomes depending on the industry or geography[cite: 7]. This research serves as a vital tool for students and job seekers who are navigating 
    an increasingly volatile economic landscape, providing them with data-driven insights rather than anecdotal evidence[cite: 10]. Furthermore, policymakers can utilize these findings to identify 
    under-compensated sectors or regions that may require targeted economic intervention or educational subsidies. The overarching goal is to transform raw labor data into a transparent map of the American 
    economy, ensuring that the relationship between labor and reward is clearly understood by all stakeholders.
    """)
    
    # Placeholder for Visual
    st.image("https://raw.githubusercontent.com/Soorej30/wage_analysis/bc4905284376f6e295d09b80bbd250c8a19b5579/labor_dynamics.png", 
             caption="Figure 1: Conceptual visualization of wage distribution across the United States.")

    # Paragraph 2: Stakeholders
    st.markdown("""
    **Stakeholders:** The impact of this research extends across a broad spectrum of economic actors, from individual workers to multi-national corporations. Job seekers and students are the primary beneficiaries, 
    as they can use these models to forecast potential earnings and make informed decisions about their educational investments[cite: 10]. Organizations and HR departments also stand to benefit by 
    benchmarking their compensation packages against national averages to remain competitive in the war for talent. On a macro level, economic researchers and government agencies like the Bureau of Labor Statistics 
    rely on these analytical interpretations to refine their reporting and understand the efficacy of labor laws. Furthermore, career counselors and academic institutions can use these insights to tailor 
    their curricula to the needs of high-growth, high-wage industries. Ultimately, anyone with a stake in the American workforce is affected by the underlying trends we aim to uncover.
    """)

    # Paragraph 3: Existing Solutions & Gaps
    st.markdown("""
    **Existing Solutions & Gaps:** Current solutions for wage estimation often rely on static tables or simplified calculators provided by sites like Glassdoor or the BLS's own basic search tools. 
    While these provide a baseline, they frequently fail to account for the interplay between multiple variables, such as how geographic location might amplify or diminish the value of a specific industry code[cite: 35, 36]. 
    Literature suggests that while many models exist for predicting income, they often ignore the "Wage Suppression Bias" found in niche occupations where small employment groups lead to data gaps. 
    Research by Smith et al. (2023) highlights that industry aggregation often hides role-specific nuances, leading to generalized predictions that don't apply to specialized technical roles. 
    Our project seeks to fill this gap by using clustering techniques to identify sub-group patterns that traditional regression might overlook[cite: 34].
    """)

    # Paragraph 4: Blueprint for the Project
    st.markdown("""
    **Blueprint for Your Project:** Our team will execute a multi-phase analytical plan starting with rigorous exploratory data analysis (EDA) to handle missing values and wage suppression[cite: 11, 32]. 
    We will then move into feature engineering, where we normalize wages by regional cost-of-living to ensure that high-salary states like California or New York are compared fairly against states with 
    lower overhead[cite: 37]. The modeling phase will involve training multiple machine learning algorithms, including Random Forests and Gradient Boosting, to predict Mean Annual Wages[cite: 8]. 
    We will also implement unsupervised learning (clustering) to see if we can discover "hidden" categories of jobs that share similar wage structures despite being in different industries[cite: 11, 34]. 
    Finally, we will develop an interactive dashboard that allows users to input their occupation and location to receive a predicted wage percentile.
    """)

    # Paragraph 5: Dataset Mention
    st.markdown("""
    **Dataset Considerations:** The primary engine of our analysis is the Occupational Employment and Wage Statistics (OEWS) dataset provided by the U.S. Bureau of Labor Statistics[cite: 13]. 
    This dataset is exceptionally robust, covering approximately 800 occupations and up to 40 variables, including mean, median, and various wage percentiles[cite: 16, 17, 25]. 
    We are also considering augmenting this with regional Consumer Price Index (CPI) data to further refine our geographic cost-of-living normalization. The public nature of the OEWS data ensures 
    transparency, though we must remain vigilant regarding suppressed values in smaller employment groups[cite: 27]. By combining these rich data sources, we believe we can build a predictive model 
    that is both accurate and ethically grounded.
    """)

# --- TAB 2: PROPOSAL OVERVIEW ---
elif tabs == "Proposal Overview":
    st.title("Project Scope & Research Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Scope Summary")
        st.write("- **Goal:** Predict and explain wage variation using ML[cite: 6].")
        st.write("- **Data:** ~800 occupations from the US BLS[cite: 13, 16].")
        st.write("- **Key Variables:** Industry Code, Location, and Wage Percentiles[cite: 21, 22, 25].")
        st.write("- **Techniques:** Regression, Clustering, and EDA[cite: 11].")

    with col2:
        st.image("https://raw.githubusercontent.com/Soorej30/wage_analysis/ccad2bf5306b1cf53ad2b9ec899f9b90502fbce1/flowchart.png", 
                 caption="Project Methodology Workflow")

    st.divider()
    
    st.header("Research Questions")
    questions = [
        "1. What is the correlation between employment level and mean annual wage across all occupations? (Descriptive)",
        "2. Which five industries currently offer the highest median wage for entry-level roles? (Descriptive)",
        "3. How do wages for the same occupation title vary significantly between different US states? (Comparative)",
        "4. Is there a statistically significant wage gap between 'Management' and 'Technical' roles within the same industry? (Comparative)",
        "5. Can we accurately predict the 90th percentile wage of an occupation based on its 10th percentile and industry code? (Predictive)",
        "6. Which geographic regions exhibit the highest 'wage-to-cost-of-living' ratio? (Comparative)",
        "7. Can clustering identify groups of occupations that are underpaid relative to their required education levels? (Predictive/Pattern)",
        "8. How has the wage gap between the 10th and 90th percentiles changed across the top 10 industries? (Descriptive)",
        "9. To what extent does 'Industry Aggregation' obscure wage differences in specialized sub-roles? (Comparative) [cite: 33]",
        "10. Can we predict if an occupation belongs to a 'High Growth' category based solely on its current wage percentiles? (Predictive)"
    ]
    for q in questions:
        st.write(q)

# --- TAB 3: TEAM ---
elif tabs == "Team":
    st.title("Meet Group 7")
    st.info("**Mission Statement:** To provide clarity in the labor market through transparent, data-driven analysis of wage structures.")
    
    team_members = [
        {"name": "Soorej S Nair", "role": "Modeling Lead", "bio": "Focused on regression analysis and predictive accuracy.", "linkedin": "https://www.linkedin.com/in/soorej-s-nair-73559470/", "github": "https://github.com/Soorej30", "img_url": "https://raw.githubusercontent.com/Soorej30/wage_analysis/c2d935bdb9a66e86f67c4f6b50face29673f0dd6/Soorej_img.jpeg"},
        {"name": "Anjana Anand", "role": "Visualization Lead", "bio": "Expert in creating intuitive and interactive dashboards.", "linkedin":"https://www.linkedin.com/in/anjana-anand-b63076398/", "github": "https://github.com/anjana5anand", "img_url": ""},
        {"name": "Nivid Pathak", "role": "Data Lead", "bio": "Specializes in data cleaning and ETL processes.", "linkedin": " ", "github": " ", "img_url": ""},
        {"name": "Karan Cheemalapati", "role": "Project Manager", "bio": "Ensures milestone alignment and documentation quality.", "linkedin": " ", "github": " ", "img_url": ""}
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