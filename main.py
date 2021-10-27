import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)
from imblearn.metrics import specificity_score
import difflib as dl
import os


# Title and description
st.title("Robustness and Sensitivity of BERT Models Predicting Alzheimer's Disease from Text")
st.markdown("Supplemantary material accompanying the following paper: Jekaterina Novikova (2021).[Robustness and Sensitivity of BERT Models Predicting Alzheimer's Disease from Text](https://arxiv.org/abs/2109.11888). \
	*In: The 7th Workshop on Noisy User-generated Text at EMNLP*, 2021.", unsafe_allow_html=True)
st.image('img/poster2.png')
st.write("[Link](https://arxiv.org/abs/2109.11888) to the high-res version of the poster.")

# Loading data
my_data = "data/df_test_all.csv"
@st.cache(persist = True)
def load_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df

df = load_data(my_data)

# Sidebar to select type and level of perturbation selection menu
st.sidebar.title("Selection Menu")
st.sidebar.markdown("Please select the type and the level of text perturbation below. <hr>", unsafe_allow_html=True)

type = st.sidebar.selectbox('Type of perturbations', ["Original / No perturbations", "Delete filled pauses", "Delete info units", "Back-translation", "Substitute with WordNet synonyms"])
level = None
iu_type = None

if type in ["Substitute with word2vec", "Substitute with WordNet synonyms"]:
	level = st.sidebar.slider('Level of perturbations:', min_value = 0.1, max_value = 0.90, step = 0.10)
elif type == "Delete info units":
	iu_type = st.sidebar.radio('Type of info units:', ["Action only", "Location only", "Object only", "Subject only"])


# select column names based on subtype of perturbations:
def select_pred_column(type, level = None, iu_type = None):
	if type == "Original / No perturbations":
		prediction = "pred_original"
	elif type == "Delete filled pauses":
		prediction = "pred_no_filled_pause"
	elif type == "Delete info units":
		if iu_type == "Action only":
			prediction = "pred_no_iu_action"
		elif iu_type == "Location only":
			prediction = "pred_no_iu_loc"
		elif iu_type == "Object only":
			prediction = "pred_no_iu_obj"
		elif iu_type == "Subject only":
			prediction = "pred_no_iu_subj"
	elif type == "Back-translation":
		prediction = "pred_back_transl"
	elif type == "Substitute with word2vec":
		lvl_str = str(level * 100)[:2]
		prediction = "pred_w2v_"+lvl_str
	elif type == "Substitute with WordNet synonyms":
		lvl_str = str(level * 100)[:2]
		prediction = "pred_wnet_"+lvl_str
	return prediction

def select_aug_column(type, level = None, iu_type = None):
	if type == "Original / No perturbations":
		augmentation = "utterances"
	elif type == "Delete filled pauses":
		augmentation = "aug_no_filled_pause"
	elif type == "Delete info units":
		if iu_type == "Action only":
			augmentation = "aug_no_iu_action"
		elif iu_type == "Location only":
			augmentation = "aug_no_iu_loc"
		elif iu_type == "Object only":
			augmentation = "aug_no_iu_obj"
		elif iu_type == "Subject only":
			augmentation = "aug_no_iu_subj"
	elif type == "Back-translation":
		augmentation = "aug_back_transl"
	elif type == "Substitute with word2vec":
		lvl_str = str(level * 100)[:2]
		augmentation = "aug_w2v_"+lvl_str
	elif type == "Substitute with WordNet synonyms":
		lvl_str = str(level * 100)[:2]
		augmentation = "aug_wnet_"+lvl_str
	return augmentation

#part I
st.header("1. Classification Performance")

st.write("The performance of the fine-tuned BERT model tested on the samples of text with applied perturbations, as selected in the Selection Menu.")

if st.button("Calculate performance"):
	acc = accuracy_score(df.label.values, df[select_pred_column(type, level, iu_type)].values)
	f1 = f1_score(df.label.values, df[select_pred_column(type, level, iu_type)].values)
	prec = precision_score(df.label.values, df[select_pred_column(type, level, iu_type)].values)
	rec = recall_score(df.label.values, df[select_pred_column(type, level, iu_type)].values)
	spec = specificity_score(df.label.values, df[select_pred_column(type, level, iu_type)].values)

	df_perf = pd.DataFrame([acc, f1, prec, rec, spec])
	df_perf.index = ["Accuracy", "F1-score", "Precision", "Recall/Sensitivity", "Specificity"]
	df_perf.columns = ["Performance"]
	st.table( df_perf.T)

#part II
st.header("2. Examples of Text Perturbations")


def text_to_code(text):
	if text == "Healthy Control (label 0)":
		code = [0]
	elif text == "Alzheimer's Disease (label 1)":
		code = [1]
	else:
		code = [0,1]
	return code

dx = st.radio('Real disease:', ["Alzheimer's Disease (label 1)", "Healthy Control (label 0)", "both"])
pred1 = st.radio('Original prediction (before text perturbation):', ["Alzheimer's Disease (label 1)", "Healthy Control (label 0)", "Don't care"])
pred2 = st.radio('Prediction after text perturbation:', ["Alzheimer's Disease (label 1)", "Healthy Control (label 0)", "Don't care"])

subject_ids = df[(df["label"].isin(text_to_code(dx))) & \
				 (df["pred_original"].isin(text_to_code(pred1))) &\
				 (df[select_pred_column(type, level, iu_type)].isin(text_to_code(pred2)))]["subject_id"]

st.write('There are', subject_ids.shape[0], 'text sample(s) that correspond to such a selection.')

if subject_ids.shape[0] > 0:
	subj_choice = st.selectbox("Select a text sample:", subject_ids)

	df_select = df[df.subject_id == subj_choice][["subject_id", "sex", "age", "label", "pred_original", select_pred_column(type, level, iu_type)]]
	df_select.age = df_select.age.astype(int)
	df_select.columns = ["SubjectID", "Sex", "Age", "Real disease label", "Original prediction", "Prediction after perturbation"]
	st.table(df_select)

	text_orig = df[df.subject_id == subj_choice]["utterances"].values[0]
	text_aug = df[df.subject_id == subj_choice][select_aug_column(type, level, iu_type)].values[0]

	words_aug = set(text_aug.replace("'"," ' ").split())
	words_orig = set(text_orig.replace("'"," ' ").split())
	s1 = text_orig.replace("'"," ' ").split()
	s2 = text_aug.replace("'"," ' ").split()


	seqmatcher = dl.SequenceMatcher(None, s1, s2, autojunk=False)
	res_orig, res_aug = [], []
	for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
	    if tag == "equal":
	        res_orig += s1[a0:a1]
	        res_aug += s2[b0:b1]
	    else:
	        res_orig += ["<span style='color:blue'> <em><b>"+" ".join(s1[a0:a1])+"</b></em></span>"]
	        res_aug += ["<span style='color:red'> <em><b>"+" ".join(s2[b0:b1])+"</b></em></span> "]

	st.write("**<span style='font-size:larger'>The original text</span>**<br>(words are coloured in blue if they were selected for perturbation):", unsafe_allow_html=True)
	st.write('<p style="padding: 1em">'+' '.join(res_orig)+'</p>', unsafe_allow_html=True)


	st.write("**<span style='font-size:larger'>The perturbed text</span>**<br>(words are coloured in red if they appeared after perturbation):", unsafe_allow_html=True)
	st.write('<p style="padding: 1em">'+' '.join(res_aug)+'</p>', unsafe_allow_html=True)

#part III
#st.header("3. Correlations")