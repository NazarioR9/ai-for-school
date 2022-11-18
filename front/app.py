from front_utils import *
from ekinox.kmeans import end_to_end_kmeans
from ekinox.ml_xgboost import end_to_end_xgboost
from ekinox.linear_model import end_to_end_linear_model
import matplotlib.pyplot as plt

PATH = 'images/'
ICON = PATH + 'crypto.jpeg'
LOGO = PATH + 'logo-ai-for-school.jpeg'
TITLE = 'AI For School'
SLOGAN = 'AI used to enhance students grade'
NAV_LIST = ['Trivial use with Linear model', 'Semi-supervised AI', 'Supervised AI with XGBoost']
DESCRIPTIONS = [
    'This method use a Linear model to analyse the relationship between a <br>\
    set of varaibles and the student actual grade. Then use the learned coefficient <br>\
    to estimate an improvability score for each student.',
    
    'This method use a semi-supervied method to group student with similar background. <br>\
    And then, within each cluster,  we estime the improvability score by comparing each <br>\
    student with the top students within the same group.', 
    
    "This approach is pretty straighforward. First, we use a model to estimate the student's grade. <br>\
    Then we compute the difference between the actual grade and the predicted grade. <br>\
    And Finally, we use that difference delta to estimate the improvability score. <br>\
    Simple put, if the predicted grade is greater than than the actual grade, <br>\
    then the student has great chance of improvability."
]

st.set_page_config(page_title=TITLE, page_icon=LOGO, layout="wide", initial_sidebar_state='expanded')
tile_placeholder = st.empty()
tile_logo = st.empty()

# Sidebar
st.sidebar.title(TITLE)
st.sidebar.image(LOGO, caption=SLOGAN)

st.sidebar.title('Navigation')
selected_page = st.sidebar.radio(
    "Please select a model/technique",
    NAV_LIST
)

# Title
title_markdown = f"<h1 style='text-align: center; color: black;'>{selected_page}</h1>"
tile_placeholder.markdown(title_markdown, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    with st.expander("Dataset snapshot"):
        st.write(dataframe.head())
    
if selected_page == NAV_LIST[0]:
    markdown_title_with_expander("Description", DESCRIPTIONS[0])
    
    if uploaded_file is not None:
        df = dataframe.copy()
        end_to_end_linear_model(df)
                
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df['FinalGrade'], df['improvability'] * 10)
        plt.xlim(df['FinalGrade'].max(), df['FinalGrade'].min())
        plt.xlabel('Final Grade')
        plt.ylabel('Improvability score')
        plt.title('Improvabilty charts')
        st.pyplot(fig)

if selected_page == NAV_LIST[1]:
    markdown_title_with_expander("Description", DESCRIPTIONS[1])
    
    if uploaded_file is not None:
        df = dataframe.copy()
        end_to_end_kmeans(df)
                
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df['FinalGrade'], df['final_gain_scale'] * 5)
        plt.xlim(df['FinalGrade'].max(), df['FinalGrade'].min())
        plt.xlabel('Final Grade')
        plt.ylabel('Improvability score')
        plt.title('Improvabilty based on students with similar background/history')
        st.pyplot(fig)

if selected_page == NAV_LIST[2]:
    markdown_title_with_expander("Description", DESCRIPTIONS[2])
    
    if uploaded_file is not None:
        df = dataframe.copy()
        df = end_to_end_xgboost(df, retrain=True)
        
        fig, ax = plt.subplots(figsize=(10,5))
        condition = df['Diff'] >= 0
        ax.scatter(df.loc[condition, 'FinalGrade'], df.loc[condition, 'Diff'], c='b')
        ax.scatter(df.loc[~condition, 'FinalGrade'], df.loc[~condition, 'Diff'], c='r')
#         ax.scatter(df['FinalGrade'], df['Diff'], c='b')
        plt.xlim(df['FinalGrade'].max(), df['FinalGrade'].min())
        plt.xlabel('Final Grade')
        plt.ylabel('Improvability score')
        plt.title('Improvabilty based on estimated grade')
        plt.legend(['greater chance of improvement', 'lower chance of improvement'])
        st.pyplot(fig)