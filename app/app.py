import base64
from math import remainder
import joblib
import pandas as pd 
import streamlit as st 
from pathlib import Path
import matplotlib.pyplot as plt 
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, \
                            precision_score, recall_score, roc_auc_score, classification_report

X_train_rus = joblib.load('./pickles/X_train_rus.pkl')
y_train_rus = joblib.load('./pickles/y_train_rus.pkl')
X_test_rus = joblib.load('./pickles/X_test_rus.pkl')
y_test_rus = joblib.load('./pickles/y_test_rus.pkl')
lr_cv = joblib.load('./pickles/lr_cv.pkl')
dc_cv = joblib.load('./pickles/dc_cv.pkl')
rf_cv = joblib.load('./pickles/rf_cv.pkl')
columns_transformer = joblib.load('./pickles/columns_transformer.pkl')

predictions = {
    "Logistic Regression": joblib.load('./pickles/lr_predictions.pkl'),
    "Decision Tree Classifier": joblib.load('./pickles/dc_predictions.pkl'),
    "Random Forest Classifier": joblib.load('./pickles/rf_predictions.pkl')
}

fpaths = {
    "Logistic Regression": './images/lr_cv.svg',
    "Decision Tree Classifier": './images/dc_cv.svg',
    "Random Forest Classifier": './images/rf_cv.svg'
}

model_instances = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree Classifier": DecisionTreeClassifier,
    "Random Forest Classifier": RandomForestClassifier
}


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img width="400" height="400" src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

st.set_page_config(layout="wide")

st.markdown(f"<h3 style='text-align: center; color: white;'>Fraudulent Job Prediction System Demo</h3>", \
        unsafe_allow_html=True)

hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

with st.expander("About the dataset"):
    st.write(Path("./README.md").read_text())

with st.sidebar:
    option1 = st.selectbox("Options", ["<SELECT>", "Serialized Models", "Tweak Parameters", "Inference"])


st.write('\n')

def plot_metrics(model):
    y_pred = predictions[model]
    col1, col2 = st.columns([3, 2])
    with col1:
        f = open(fpaths[model], 'r')
        lines = f.readlines()
        line_string=''.join(lines)
        render_svg(line_string)
    with col2:
            st.write('\n')
            st.write('\n')
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            st.table(
                pd.DataFrame({
                    "Metric": ["Accuracy", "Balanced Accuracy", "ROC AUC Score", "F1 Score", "Precision", "Recall"],
                    "Value": [
                    accuracy_score(y_test_rus, y_pred), 
                    balanced_accuracy_score(y_test_rus, y_pred),
                    roc_auc_score(y_test_rus, y_pred),
                    f1_score(y_test_rus, y_pred),
                    precision_score(y_test_rus, y_pred),
                    recall_score(y_test_rus, y_pred)
                    ]
                })
            )
def plot_metrics_tweaked(model, features, **params):
    X_train_rus1 = columns_transformer.fit_transform(X_train_rus[features])
    X_test_rus1 = columns_transformer.transform(X_test_rus[features])
    m = model(**params)
    m.fit(X_train_rus1, y_train_rus)
    y_pred = m.predict(X_test_rus1)
    st.markdown(f"<h5 style='text-align: center; color: white;'>Classification Performances</h5>", \
        unsafe_allow_html=True)
    st.table(classification_report(y_test_rus, y_pred, output_dict=True))
    
def inference(df):
    pred = int(lr_cv.predict(df))
    dc_map = {
        0: "Real",
        1: "Fake"
    }
    st.markdown(f"<h5 style='text-align: center; color: red;'>Prediction: {dc_map[pred]}</h5>", \
        unsafe_allow_html=True)

if option1 == "Serialized Models":
    chosen_model = st.selectbox("Choose Model:", list(fpaths.keys()))
    plot_metrics(chosen_model)

elif option1 == "Tweak Parameters":
    chosen_model = st.sidebar.selectbox("Choose Model:", list(fpaths.keys()))
    chosen_model_instance = model_instances[chosen_model]
    form = st.sidebar.form("Form")
    features = form.multiselect("Choose Features", list(X_test_rus.columns), ["title", "function", "cleaned_text", ])
    form.write("Hyperparameters")
    if chosen_model == "Logistic Regression":
        c = form.slider("C", 10, 60, 50, 10)
        solver = form.selectbox("solver", ["newton-cg", "liblinear", "lbfgs"])
        multi_class = form.selectbox("multi_class", ["multinomial", "auto", "ovr"]) 
        form.form_submit_button(label="Submit", on_click=plot_metrics_tweaked(chosen_model_instance, features, C=c, solver=solver, multi_class=multi_class)) 
   
    elif chosen_model == "Decision Tree Classifier":
        criterion = form.selectbox("criterion", ["gini", "entropy", "log_loss"])
        splitter = form.selectbox("splitter", ["best", "random"])
        max_depth = form.slider("max_depth", 10, 50, 30, 10) 
        ccp_alpha = form.slider("ccp_alpha", 0.001, 0.1, 0.01, 0.05)     
        form.form_submit_button(label="Submit", on_click=plot_metrics_tweaked(chosen_model_instance, features, \
            max_depth=max_depth, criterion=criterion, splitter=splitter, ccp_alpha=ccp_alpha)) 

    else:
        criterion = form.selectbox("criterion", ["gini", "entropy", "log_loss"])
        max_depth = form.slider("max_depth", 10, 50, 30, 10) 
        min_samples_leaf = form.slider("min_samples_leaf", 2, 10, 5, 3) 
        ccp_alpha = form.slider("ccp_alpha", 0.001, 0.1, 0.01, 0.05)  
        form.form_submit_button(label="Submit", on_click=plot_metrics_tweaked(chosen_model_instance, features, \
            max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha))

elif option1 == "Inference":
    st.markdown(f"<h5 style='text-align: center; color: white;'>Sample Inference</h5>", \
        unsafe_allow_html=True)
    st.write('\n')
    form = st.form("Form2")
    location_isna = form.checkbox("location_isna", False)
    salary_range_isna = form.checkbox("salary_range_isna", False)
    description_isna = form.checkbox("description_isna", False)
    requirements_isna = form.checkbox("requirements_isna", False)
    benefits_isna = form.checkbox("benefits_isna", False)
    company_profile_isna = form.checkbox("company_profile_isna", False)
    function_isna = form.checkbox("funcion_isna", False)
    has_company_logo = float(form.checkbox("has_company_logo", False))
    title = form.text_input("title", max_chars=20)
    function = form.text_input("function")
    cleaned_text = form.text_input("cleaned_text", max_chars=200)
    dc = {
        "location_isna": location_isna,
        "salary_range_isna": salary_range_isna,
        "description_isna": description_isna,
        "requirements_isna": requirements_isna,
        "benefits_isna": benefits_isna,
        "company_profile_isna": company_profile_isna,
        "function_isna": function_isna,
        "has_company_logo": has_company_logo,
        "title": title,
        "function": function,
        "cleaned_text": cleaned_text

    }
    df1 = pd.DataFrame(dc, index=[0])
    form.form_submit_button(label="Submit", on_click=inference(df1))

else:
    pass





    
