import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Recommender", layout="wide")

def set_value(key):
    st.session_state[key] = st.session_state["key_" + key]

def set_status(status):
    st.session_state["status"] = status

STATE_KEYS_VALS = [
    ("selected_movie_count", 0),  # main part
    ("added_movie_ids", []),  # main part
    ("status", False),
    ("clicked", False),
    ("input_len", 20),  # sidebar
    ("input_book", ""),  # sidebar
    ("years", (1990, 2010)),  # sidebar

]
for k, v in STATE_KEYS_VALS:
    if k not in st.session_state:
        st.session_state[k] = v


#############################################################
###                  Define Side-bar View                 ###
#############################################################
st.sidebar.title("Setting")

selected_item = st.sidebar.radio('책 평가를 남긴적 있으십니까?', ("네","아니오"))


user = st.sidebar.number_input(
    "사용자 아이디를 입력하세요",
    format="%i",
    min_value=8,
    max_value=278854,
    # value=int(st.session_state["input_len"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("input_len",),
    key="key_input_len",
)
if selected_item == "아니오":
    user = False

book = st.sidebar.text_input(
    "책 아이디를 입력하세요",
    # value=int(st.session_state["input_len"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("input_book",),
    key="key_input_book",
)

st.sidebar.button(
    "START",
    on_click=set_status,
    args=(True,),
    disabled=st.session_state["status"],
)




#############################################################
###                    Define Main View                   ###
#############################################################

st.title("books Recomender with Streamlit")

users_df = pd.read_csv("data/users.csv")
books_df = pd.read_csv("data/books.csv")
train_df = pd.read_csv("data/total_ratings.csv")
train_df = train_df.merge(users_df, how='left', on='user_id').merge(books_df, how='left', on='isbn')


if user:
    if len(train_df[train_df["user_id"]==user]) == 0:
        st.write("평가 기록이 없습니다.")
    else:
        st.write(train_df[train_df["user_id"]==user])
else:
    st.write("조회할 수 없습니다!")

if st.session_state["status"]:
    st.write("평점을 예측하겠습니다!!")
    with st.spinner("wait for it..."):
        
