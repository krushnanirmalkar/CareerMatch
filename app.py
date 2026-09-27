import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CareerMatch",
    page_icon="🎯",
    layout="centered"
)

# --------------------------------------------------
# Load Model and Encoders
# --------------------------------------------------

MODEL_DIR = "model"

@st.cache_resource
def load_artifacts():
    final_knn = joblib.load(
        os.path.join(MODEL_DIR, "final_knn.pkl")
    )

    encoder = joblib.load(
        os.path.join(MODEL_DIR, "encoder.pkl")
    )

    skill_encoder = joblib.load(
        os.path.join(MODEL_DIR, "skill_encoder.pkl")
    )

    feature_columns = joblib.load(
        os.path.join(MODEL_DIR, "feature_columns.pkl")
    )

    return (
        final_knn,
        encoder,
        skill_encoder,
        feature_columns
    )


try:
    final_knn, encoder, skill_encoder, feature_columns = load_artifacts()

except Exception as e:
    st.error(
        "Unable to load the trained model files. "
        "Make sure all .pkl files are inside the model folder."
    )

    st.exception(e)
    st.stop()


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

categorical_columns = [
    "Education_Level",
    "Specialization",
    "Interests"
]


def prepare_student_input(
    education,
    specialization,
    interest,
    skills
):
    """
    Convert user input into the same encoded format
    used while training the KNN model.
    """

    # Categorical input
    categorical_input = pd.DataFrame({
        "Education_Level": [education],
        "Specialization": [specialization],
        "Interests": [interest]
    })

    # One-Hot Encoding
    categorical_encoded = encoder.transform(
        categorical_input
    )

    categorical_df = pd.DataFrame(
        categorical_encoded,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    # Multi-label encoding for skills
    skills_encoded = skill_encoder.transform(
        [skills]
    )

    skills_df = pd.DataFrame(
        skills_encoded,
        columns=[
            f"Skill_{skill}"
            for skill in skill_encoder.classes_
        ]
    )

    # Combine categorical and skill features
    student_encoded = pd.concat(
        [
            categorical_df,
            skills_df
        ],
        axis=1
    )

    # Ensure exact training feature order
    student_encoded = student_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return student_encoded


def get_top3_recommendations(
    model,
    student_input
):
    """
    Return Top-3 career recommendations
    along with KNN recommendation scores.
    """

    probabilities = model.predict_proba(
        student_input
    )[0]

    top3_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    careers = model.classes_[top3_indices]

    scores = probabilities[top3_indices]

    return list(
        zip(careers, scores)
    )


# --------------------------------------------------
# Extract Available Options From Encoders
# --------------------------------------------------

education_options = list(
    encoder.categories_[0]
)

specialization_options = list(
    encoder.categories_[1]
)

interest_options = list(
    encoder.categories_[2]
)

skill_options = list(
    skill_encoder.classes_
)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🎯 CareerMatch")

st.subheader(
    "Skill and Interest Based Career Role Recommendation Using KNN"
)

st.write(
    """
    CareerMatch analyzes your educational background,
    specialization, interests, and skills to recommend
    the **Top 3 career roles** that are most similar to
    profiles present in the training dataset.
    """
)

st.divider()


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.header("Enter Your Profile")

education = st.selectbox(
    "Education Level",
    education_options
)

specialization = st.selectbox(
    "Specialization",
    specialization_options
)

interest = st.selectbox(
    "Primary Interest",
    interest_options
)

skills = st.multiselect(
    "Select Your Skills",
    skill_options
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "Get Career Recommendations",
    type="primary",
    use_container_width=True
):

    if len(skills) == 0:
        st.warning(
            "Please select at least one skill."
        )

    else:

        student_input = prepare_student_input(
            education=education,
            specialization=specialization,
            interest=interest,
            skills=skills
        )

        recommendations = get_top3_recommendations(
            final_knn,
            student_input
        )

        st.success(
            "Career recommendations generated successfully."
        )

        st.subheader("Your Top Career Recommendations")

        # --------------------------------------------------
        # First Recommendation
        # --------------------------------------------------

        career_1, score_1 = recommendations[0]

        st.markdown(
            f"""
            ### 🥇 {career_1}

            **Recommendation Score:**  
            `{score_1 * 100:.2f}%`
            """
        )

        st.progress(
            float(score_1)
        )

        # --------------------------------------------------
        # Second Recommendation
        # --------------------------------------------------

        career_2, score_2 = recommendations[1]

        st.markdown(
            f"""
            ### 🥈 {career_2}

            **Recommendation Score:**  
            `{score_2 * 100:.2f}%`
            """
        )

        st.progress(
            float(score_2)
        )

        # --------------------------------------------------
        # Third Recommendation
        # --------------------------------------------------

        career_3, score_3 = recommendations[2]

        st.markdown(
            f"""
            ### 🥉 {career_3}

            **Recommendation Score:**  
            `{score_3 * 100:.2f}%`
            """
        )

        st.progress(
            float(score_3)
        )

        # --------------------------------------------------
        # Summary Table
        # --------------------------------------------------

        st.subheader("Recommendation Summary")

        result_df = pd.DataFrame({
            "Rank": [
                "1",
                "2",
                "3"
            ],
            "Career": [
                career_1,
                career_2,
                career_3
            ],
            "Recommendation Score": [
                f"{score_1 * 100:.2f}%",
                f"{score_2 * 100:.2f}%",
                f"{score_3 * 100:.2f}%"
            ]
        })

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            """
            The recommendation score represents the proportion
            of nearby profiles in the KNN model that support
            each career recommendation.

            It should not be interpreted as a guarantee of
            career success or suitability.
            """
        )


# --------------------------------------------------
# Project Information
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ How CareerMatch Works"):

    st.write(
        """
        CareerMatch uses the **K-Nearest Neighbors (KNN)**
        machine learning algorithm.

        The system converts the user's educational background,
        specialization, interests, and skills into numerical
        features.

        KNN then finds the most similar student profiles
        from the training dataset.

        The career roles of those neighboring profiles vote
        for the final recommendations.

        Instead of returning only one career, CareerMatch
        returns the **Top 3 career recommendations** because
        similar skill profiles may be suitable for multiple
        career paths.
        """
    )


with st.expander("Model Information"):

    st.write(
        """
        **Algorithm:** K-Nearest Neighbors (KNN)

        **Problem Type:** Multi-Class Classification

        **Input Features:**
        - Education Level
        - Specialization
        - Skills
        - Interests

        **Output:** Top 3 Recommended Career Roles
        """
    )


st.caption(
    "CareerMatch • Foundations of AIML Mini Project"
)