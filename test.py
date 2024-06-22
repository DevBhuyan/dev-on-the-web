#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 03:17:23 2024

@author: dev
"""


from chat import ask_about_me
import streamlit as st
from streamlit import session_state as ss
from tictactoe import start


def for_everyone():
    st.title("Oops! Looks like you're early")

    for _ in range(3):
        st.title(" ")

    st.subheader("Fret not! Here's something fun to play")

    _, col2, _ = st.columns(3)

    with col2:
        if st.button("Click here to play!"):
            ss.page = 'game'
            st.rerun()

    ip = st.chat_input(
        "Enter the passkey if you want to access this secret place")
    if ip == "0000":
        ss.page = 'owner'
        st.rerun()


def tic_tac_toe():
    st.title("The game of Tic-Tac-Toe")
    st.subheader("But with an RL powered Bot")
    start()


def main_page():
    st.title("Hi! I'm Dev 👋")

    st.write("I'm an Electronics Undergrad with a passion for Machine Learning")

    ip = st.chat_input("Ask me anything")

    if ip:
        st.text_area("", ask_about_me(ip))


if __name__ == "__main__":
    if 'page' not in ss:
        ss['page'] = 'everyone'
    if ss.page == 'everyone':
        for_everyone()
    if ss.page == 'game':
        tic_tac_toe()
    if ss.page == 'owner':
        main_page()
