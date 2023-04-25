#!/usr/bin/env python3
#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#

import sys
import timeit

import emoji

import PIL as pillow

import streamlit as st

#----------------------------------------------------------------------------------------------------------------------#

import module.regression_kaggle_insurance
import module.regression_random_linear
import module.classification_random_binary

#----------------------------------------------------------------------------------------------------------------------#

def main():

  if 'verbose' not in st.session_state:
    st.session_state['verbose'] = False

  ########################

  st.set_page_config(layout='wide', page_title='Tensor example')

  with open('style.css', encoding='utf-8') as file:
    st.markdown('<style>' + file.read() + '</style>', unsafe_allow_html=True)

  ########################

  st.title('TensorFlow', 'title')
  st.markdown('##### {} laboratory'.format(emoji.emojize(':lab_coat:')))

  st.image(pillow.Image.open('tensorflow.webp'), width=128)

  st.button('Rerun')

  ########################

  if st.sidebar.button(emoji.emojize(':speech_balloon:')):
    st.session_state['verbose'] = not st.session_state['verbose']

  ########################

  verbose =  st.session_state['verbose']
  if verbose:
    st.info('verbose = {}'.format(verbose), icon=emoji.emojize(':speech_balloon:'))

  ########################

  timer = timeit.default_timer()

  ########################

  options = {
    '---': sys.modules[__name__],
    'regression (random linear)': module.regression_random_linear,
    'regression (kaggle insurance)': module.regression_kaggle_insurance,
    'classification (random binary)': module.classification_random_binary
  }

  option = options[st.sidebar.selectbox('module', options.keys())]
  option.run()

  ########################

  st.markdown('---')

  st.markdown('##### -')
  st.markdown('##### {} in {:.3f} second(s)'.format(emoji.emojize(':rocket:'), timeit.default_timer() - timer))

  del timer

#----------------------------------------------------------------------------------------------------------------------#

def run():

  st.markdown('---')
  st.markdown('#### {} Dependancy'.format(emoji.emojize(':x-ray:')))

  st.table({'module': ['python', 'streamlit'], 'version': [sys.version.split(maxsplit=1)[0], st.__version__]})

#----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
  main()

#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#
