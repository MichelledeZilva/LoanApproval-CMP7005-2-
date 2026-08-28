import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "Models"
DATA_DIR = BASE_DIR / "Data"

DT_MODEL_PATH = MODEL_DIR / "dt_final_model.pkl"
LR_MODEL_PATH = MODEL_DIR / "logistic_final_model.pkl"

MERGED_DATA_PATH = DATA_DIR / "merged_df.csv"


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b0f17;
        color: #f5f7fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #aab4c3;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    section[data-testid="stSidebar"] {
        background-color: #101620;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 17px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    dt_model = joblib.load(DT_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH)

    return dt_model, lr_model


try:

    dt_final_model, logistic_final_model = load_models()

except Exception as e:

    st.error("Unable to load the trained models.")

    st.write("Please check that these files exist:")

    st.code(str(DT_MODEL_PATH))
    st.code(str(LR_MODEL_PATH))

    st.exception(e)

    st.stop()


# ============================================================
# LOAD FINAL MERGED DATASET
# ============================================================

@st.cache_data
def load_merged_data():

    return pd.read_csv(MERGED_DATA_PATH)


try:

    merged_df = load_merged_data()

except Exception:

    merged_df = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏦 Loan Approval Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning application for loan approval prediction'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🏦 Loan Approval")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Modelling & Prediction",
        "📊 Data Overview",
        "📈 EDA"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Available Models")

st.sidebar.write("🌳 Decision Tree")
st.sidebar.write("📈 Logistic Regression")

st.sidebar.markdown("---")

st.sidebar.caption("Programming for Data Analysis (CMP 7005)")
st.sidebar.caption("Task 4 – Application Development")


# ============================================================
# PAGE 1
# MODELLING & PREDICTION
# ============================================================

if page == "🔮 Modelling & Prediction":

    st.markdown(
        '<div class="section-title">'
        '🔮 Modelling & Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter applicant and loan information and select "
        "a machine learning model for prediction."
    )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.markdown("### 🤖 Model Selection")

    model_choice = st.selectbox(
        "Select a prediction model",
        [
            "Decision Tree",
            "Logistic Regression"
        ]
    )

    if model_choice == "Decision Tree":

        st.info(
            "Decision Tree selected. This uses the final "
            "optimized Decision Tree model."
        )

    else:

        st.info(
            "Logistic Regression selected. This uses the final "
            "optimized Logistic Regression model."
        )


    # ========================================================
    # APPLICANT INFORMATION
    # ========================================================

    st.markdown("### 👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

    with col2:

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Employed",
                "Self-Employed",
                "Unemployed"
            ]
        )

    with col3:

        education_level = st.selectbox(
            "Education Level",
            [
                "Master",
                "Associate",
                "Bachelor",
                "High School",
                "Doctorate"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Single",
                "Divorced",
                "Widowed",
                "Unknown"
            ]
        )

    with col2:

        number_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    with col3:

        experience = st.number_input(
            "Experience (Years)",
            min_value=0,
            max_value=60,
            value=10,
            step=1
        )


    col1, col2 = st.columns(2)

    with col1:

        job_tenure = st.number_input(
            "Job Tenure (Years)",
            min_value=0,
            max_value=60,
            value=5,
            step=1
        )

    with col2:

        home_ownership_status = st.selectbox(
            "Home Ownership Status",
            [
                "Own",
                "Mortgage",
                "Rent",
                "Other"
            ]
        )


    # ========================================================
    # FINANCIAL INFORMATION
    # ========================================================

    st.markdown("### 💰 Financial Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=1200000.0,
            step=10000.0
        )

    with col2:

        checking_account_balance = st.number_input(
            "Checking Account Balance",
            min_value=0.0,
            value=150000.0,
            step=10000.0
        )

    with col3:

        savings_account_balance = st.number_input(
            "Savings Account Balance",
            min_value=0.0,
            value=250000.0,
            step=10000.0
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        total_assets = st.number_input(
            "Total Assets",
            min_value=0.0,
            value=1500000.0,
            step=10000.0
        )

    with col2:

        total_liabilities = st.number_input(
            "Total Liabilities",
            min_value=0.0,
            value=300000.0,
            step=10000.0
        )

    with col3:

        net_worth = st.number_input(
            "Net Worth",
            value=1000000.0,
            step=10000.0
        )


    # ========================================================
    # CREDIT INFORMATION
    # ========================================================

    st.markdown("### 💳 Credit Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=0,
            max_value=1000,
            value=700,
            step=1
        )

    with col2:

        risk_score_available = st.radio(
            "Is Risk Score available?",
            [
                "Yes",
                "No"
            ],
            horizontal=True
        )

    with col3:

        number_of_open_credit_lines = st.number_input(
            "Number of Open Credit Lines",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )


    # ========================================================
    # RISK SCORE
    # ========================================================

    if risk_score_available == "Yes":

        risk_score = st.number_input(
            "Risk Score",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        risk_score_missing = False

    else:

        risk_score = 51.0

        risk_score_missing = True

        st.info(
            "Risk Score is treated as missing. "
            "The dataset median value of 51.0 is used."
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        length_of_credit_history = st.number_input(
            "Length of Credit History",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )

    with col2:

        number_of_credit_inquiries = st.number_input(
            "Number of Credit Inquiries",
            min_value=0,
            max_value=100,
            value=2,
            step=1
        )

    with col3:

        payment_history = st.number_input(
            "Payment History",
            min_value=0,
            max_value=100,
            value=30,
            step=1
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        credit_card_utilization_rate = st.number_input(
            "Credit Card Utilization Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=0.1
        )

    with col2:

        previous_loan_defaults = st.selectbox(
            "Previous Loan Defaults",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with col3:

        bankruptcy_history = st.selectbox(
            "Bankruptcy History",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )


    # ========================================================
    # LOAN INFORMATION
    # ========================================================

    st.markdown("### 🏦 Loan Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=500000.0,
            step=10000.0
        )

    with col2:

        loan_duration = st.number_input(
            "Loan Duration",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

    with col3:

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.01,
            max_value=100.0,
            value=18.0,
            step=0.01
        )


    col1, col2 = st.columns(2)

    with col1:

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Home",
                "Debt Consolidation",
                "Education",
                "Other",
                "Auto"
            ]
        )

    with col2:

        monthly_debt_payments = st.number_input(
            "Monthly Debt Payments",
            min_value=0.0,
            value=20000.0,
            step=1000.0
        )


    # ========================================================
    # CALCULATED FEATURES
    # ========================================================

    monthly_income = annual_income / 12

    monthly_rate = interest_rate / (100 * 12)

    if monthly_rate > 0:

        monthly_loan_payment = (
            loan_amount
            * monthly_rate
            * (1 + monthly_rate) ** loan_duration
        ) / (
            (1 + monthly_rate) ** loan_duration - 1
        )

    else:

        monthly_loan_payment = (
            loan_amount / loan_duration
        )


    if monthly_income > 0:

        total_debt_to_income_ratio = (
            (
                monthly_loan_payment
                + monthly_debt_payments
            )
            * 100
            / monthly_income
        )

    else:

        total_debt_to_income_ratio = 0.0


    # ========================================================
    # CALCULATED VALUES DISPLAY
    # ========================================================

    st.markdown("### 🧮 Calculated Values")

    calc1, calc2, calc3 = st.columns(3)

    with calc1:

        st.metric(
            "Monthly Income",
            f"{monthly_income:,.2f}"
        )

    with calc2:

        st.metric(
            "Monthly Loan Payment",
            f"{monthly_loan_payment:,.2f}"
        )

    with calc3:

        st.metric(
            "Debt-to-Income Ratio",
            f"{total_debt_to_income_ratio:.2f}%"
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Loan Approval"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        input_data = pd.DataFrame([{

            "AnnualIncome":
                annual_income,

            "EmploymentStatus":
                employment_status,

            "LoanDuration":
                loan_duration,

            "NumberOfDependents":
                number_of_dependents,

            "MonthlyDebtPayments":
                monthly_debt_payments,

            "NumberOfOpenCreditLines":
                number_of_open_credit_lines,

            "BankruptcyHistory":
                bankruptcy_history,

            "PreviousLoanDefaults":
                previous_loan_defaults,

            "LengthOfCreditHistory":
                length_of_credit_history,

            "CheckingAccountBalance":
                checking_account_balance,

            "TotalLiabilities":
                total_liabilities,

            "NetWorth":
                net_worth,

            "Age":
                age,

            "CreditScore":
                credit_score,

            "EducationLevel":
                education_level,

            "LoanAmount":
                loan_amount,

            "MaritalStatus":
                marital_status,

            "HomeOwnershipStatus":
                home_ownership_status,

            "CreditCardUtilizationRate":
                credit_card_utilization_rate,

            "NumberOfCreditInquiries":
                number_of_credit_inquiries,

            "LoanPurpose":
                loan_purpose,

            "PaymentHistory":
                payment_history,

            "SavingsAccountBalance":
                savings_account_balance,

            "TotalAssets":
                total_assets,

            "JobTenure":
                job_tenure,

            "InterestRate":
                interest_rate,

            "RiskScore":
                risk_score,

            "Experience":
                experience,

            "RiskScoreMissing":
                risk_score_missing,

            "MonthlyLoanPayment":
                monthly_loan_payment,

            "TotalDebtToIncomeRatio":
                total_debt_to_income_ratio
        }])


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            if model_choice == "Decision Tree":

                prediction = dt_final_model.predict(
                    input_data
                )

                probability = dt_final_model.predict_proba(
                    input_data
                )[0][1]

            else:

                prediction = logistic_final_model.predict(
                    input_data
                )

                probability = logistic_final_model.predict_proba(
                    input_data
                )[0][1]


            probability_percentage = probability * 100


            if prediction[0] == 1:

                result_text = "Approved"

            else:

                result_text = "Not Approved"


            # =================================================
            # RESULT
            # =================================================

            st.markdown("---")

            st.markdown("## 📋 Prediction Result")


            if prediction[0] == 1:

                st.success("✅ LOAN APPROVED")

            else:

                st.error("❌ LOAN NOT APPROVED")


            result_col1, result_col2, result_col3 = st.columns(3)


            with result_col1:

                st.metric(
                    "Selected Model",
                    model_choice
                )


            with result_col2:

                st.metric(
                    "Prediction",
                    result_text
                )


            with result_col3:

                st.metric(
                    "Approval Probability",
                    f"{probability_percentage:.2f}%"
                )


            # =================================================
            # DETAILS
            # =================================================

            st.markdown("### 📋 Prediction Details")

            detail1, detail2 = st.columns(2)

            with detail1:

                st.write(
                    f"**Selected Model:** {model_choice}"
                )

                st.write(
                    f"**Prediction:** {result_text}"
                )

            with detail2:

                st.write(
                    f"**Approval Probability:** "
                    f"{probability_percentage:.2f}%"
                )

                if risk_score_missing:

                    st.write(
                        "**Risk Score:** 51.0 "
                        "(dataset median)"
                    )

                else:

                    st.write(
                        f"**Risk Score:** {risk_score:.1f}"
                    )



        except Exception as e:

            st.error(
                "An error occurred while generating the prediction."
            )

            st.exception(e)


# ============================================================
# PAGE 2
# DATA OVERVIEW
# ============================================================

elif page == "📊 Data Overview":

    st.markdown("## 📊 Data Overview")

    st.write(
        "Overview of the final merged loan approval dataset."
    )


    if merged_df is None:

        st.error(
            "The merged dataset could not be loaded."
        )

        st.write(
            "Please make sure this file exists:"
        )

        st.code(
            str(MERGED_DATA_PATH)
        )

    else:

        # ----------------------------------------------------
        # Dataset metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Records",
                f"{merged_df.shape[0]:,}"
            )

        with col2:

            st.metric(
                "Columns",
                merged_df.shape[1]
            )

        with col3:

            st.metric(
                "Missing Values",
                int(
                    merged_df.isna()
                    .sum()
                    .sum()
                )
            )

        with col4:

            st.metric(
                "Duplicate Rows",
                int(
                    merged_df.duplicated()
                    .sum()
                )
            )


        # ----------------------------------------------------
        # Dataset preview
        # ----------------------------------------------------

        st.markdown("### 📋 Dataset Preview")

        st.dataframe(
            merged_df.head(10),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Data types
        # ----------------------------------------------------

        st.markdown("### 🔤 Data Types")

        dtype_df = pd.DataFrame(
            {
                "Feature":
                    merged_df.columns,

                "Data Type":
                    merged_df.dtypes.astype(str).values,

                "Missing Values":
                    merged_df.isna().sum().values,

                "Unique Values":
                    [
                        merged_df[column].nunique()
                        for column in merged_df.columns
                    ]
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Target information
        # ----------------------------------------------------

        if "LoanApproved" in merged_df.columns:

            st.markdown(
                "### 🎯 Loan Approval Summary"
            )

            approved = int(
                (
                    merged_df["LoanApproved"] == 1
                ).sum()
            )

            not_approved = int(
                (
                    merged_df["LoanApproved"] == 0
                ).sum()
            )

            target_col1, target_col2 = st.columns(2)

            with target_col1:

                st.metric(
                    "Approved",
                    f"{approved:,}"
                )

            with target_col2:

                st.metric(
                    "Not Approved",
                    f"{not_approved:,}"
                )


# ============================================================
# PAGE 3
# EDA
# ============================================================

elif page == "📈 EDA":

    st.markdown(
        "## 📈 Exploratory Data Analysis"
    )

    st.write(
        "Interactive exploration of the final merged dataset."
    )


    if merged_df is None:

        st.error(
            "The merged dataset could not be loaded."
        )

        st.code(
            str(MERGED_DATA_PATH)
        )

    else:

        # ====================================================
        # LOAN APPROVAL DISTRIBUTION
        # ====================================================

        if "LoanApproved" in merged_df.columns:

            st.markdown(
                "### 🎯 Loan Approval Distribution"
            )

            target_counts = (
                merged_df["LoanApproved"]
                .value_counts()
                .rename(
                    index={
                        0: "Not Approved",
                        1: "Approved"
                    }
                )
            )

            st.bar_chart(
                target_counts
            )


        # ====================================================
        # NUMERICAL FEATURE ANALYSIS
        # ====================================================

        st.markdown(
            "### 🔢 Numerical Feature Analysis"
        )

        numerical_columns = (
            merged_df
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )


        if "LoanApproved" in numerical_columns:

            numerical_columns.remove(
                "LoanApproved"
            )


        if numerical_columns:

            selected_numeric = st.selectbox(
                "Select a numerical feature",
                numerical_columns
            )


            st.markdown(
                "#### Summary Statistics"
            )

            st.dataframe(
                merged_df[
                    selected_numeric
                ]
                .describe()
                .to_frame(),
                use_container_width=True
            )


            st.markdown(
                "#### Mean Value by Loan Approval"
            )

            if "LoanApproved" in merged_df.columns:

                mean_by_target = (
                    merged_df
                    .groupby(
                        "LoanApproved"
                    )[selected_numeric]
                    .mean()
                    .rename(
                        index={
                            0: "Not Approved",
                            1: "Approved"
                        }
                    )
                )

                st.bar_chart(
                    mean_by_target
                )


        # ====================================================
        # CATEGORICAL FEATURE ANALYSIS
        # ====================================================

        st.markdown(
            "### 🔤 Categorical Feature Analysis"
        )

        categorical_columns = (
            merged_df
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "bool"
                ]
            )
            .columns
            .tolist()
        )


        if categorical_columns:

            selected_categorical = st.selectbox(
                "Select a categorical feature",
                categorical_columns
            )


            category_counts = (
                merged_df[
                    selected_categorical
                ]
                .value_counts()
            )


            st.markdown(
                "#### Category Distribution"
            )

            st.bar_chart(
                category_counts
            )


        # ====================================================
        # RISK SCORE ANALYSIS
        # ====================================================

        if "RiskScore" in merged_df.columns:

            st.markdown(
                "### ⚠️ Risk Score Analysis"
            )

            risk_col1, risk_col2, risk_col3 = st.columns(3)

            with risk_col1:

                st.metric(
                    "Mean Risk Score",
                    f"{merged_df['RiskScore'].mean():.2f}"
                )

            with risk_col2:

                st.metric(
                    "Median Risk Score",
                    f"{merged_df['RiskScore'].median():.2f}"
                )

            with risk_col3:

                st.metric(
                    "Maximum Risk Score",
                    f"{merged_df['RiskScore'].max():.2f}"
                )


        # ====================================================
        # CORRELATION ANALYSIS
        # ====================================================

        st.markdown(
            "### 🔗 Correlation with Loan Approval"
        )

        if "LoanApproved" in merged_df.columns:

            correlation_columns = (
                merged_df
                .select_dtypes(
                    include="number"
                )
                .columns
                .tolist()
            )

            correlation_data = (
                merged_df[
                    correlation_columns
                ]
                .corr()["LoanApproved"]
                .drop("LoanApproved")
                .sort_values(
                    key=abs,
                    ascending=False
                )
            )

            st.dataframe(
                correlation_data
                .to_frame(
                    name="Correlation"
                ),
                use_container_width=True
            )
