#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#

import sys

import emoji

import numpy as np

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st

import tensorflow as tf

#----------------------------------------------------------------------------------------------------------------------#

HEAD_SIZE = 10

#----------------------------------------------------------------------------------------------------------------------#

def run():

  verbose = st.session_state['verbose']

  ########################

  plt.style.use('dark_background')

  palette = mpl.cm.get_cmap('Pastel2').colors

  plt.rcParams['image.cmap'] = 'Pastel2'
  plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)

  plt.rcParams['figure.constrained_layout.use'] = True

  ########################

  st.sidebar.markdown('---')

  height = st.sidebar.select_slider('height', [2**number for number in range(3, 8)], 16)
  reign = st.sidebar.select_slider('reign', range(10, 90, 10), 20)

  st.sidebar.markdown('---')

  ########################

  st.markdown('---')
  st.markdown('#### {} Dataset'.format(emoji.emojize(':jar:')))

  dataset = pd.read_csv('module/kaggle/insurance.csv')

  st.markdown('{} records'.format(dataset.shape[0]))
  if verbose:
    st.info('dataset = {} {}'.format(dataset.shape, dataset.dtypes.values), icon=emoji.emojize(':speech_balloon:'))

  st.table(dataset.head(HEAD_SIZE))

  ########################

  features = pd.get_dummies(dataset.drop('charges', axis=1), dtype=np.int64)
  features['bmi'] = features['bmi'].apply(round)

  st.write('{} features'.format(features.shape[1]))
  if verbose:
    st.info('features = {} {}'.format(features.shape, features.dtypes.values),
            icon=emoji.emojize(':speech_balloon:'))

  st.table(features.head(HEAD_SIZE))

  ########################

  targets = pd.DataFrame()
  targets['charges'] = dataset['charges'].apply(round)

  st.write('{} targets'.format(targets.shape[1]))
  if verbose:
    st.info('targets = {} {}'.format(targets.shape, targets.dtypes.values), icon=emoji.emojize(':speech_balloon:'))

  st.table(targets.head(HEAD_SIZE))

  ########################

  st.markdown('---')
  st.markdown('#### {} Model'.format(emoji.emojize(':abacus:')))

  st.markdown('{} epoches'.format(reign))

  model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(height, activation='relu', name='input'),
     tf.keras.layers.Dense(1, name='output')])

  if verbose:
    st.info('model = {} [{}]'.format(model, model.dtype), icon=emoji.emojize(':speech_balloon:'))

  model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

  ########################

  frame = st.empty()

  losses = []
  for number in range(reign):

    history = model.fit(features, targets, verbose=0).history

    losses.append(round(history['loss'][0]))

    figure, axe = plt.subplots(figsize=(20, 6))
    axe.set_xlabel('epoch')
    axe.set_ylabel('loss')
    axe.set_xlim([-0.5, reign - 0.5])
    axe.set_xticks(range(0, reign))

    axe.plot(range(0, len(losses)), losses, '-o', c=palette[2], linewidth=3.0)
    frame.pyplot(figure)

    plt.close(figure)
    del figure, axe

  ########################

  st.markdown('loss = {}'.format(losses[-1]))

  with st.expander(emoji.emojize(':information:')):
    st.table({'loss': losses})

  ########################

  st.markdown('---')
  st.markdown('#### {} Prediction'.format(emoji.emojize(':bullseye:')))

  results = pd.DataFrame(model.predict(features, verbose=None), columns=['charges'])
  results['charges'] = results['charges'].apply(round) #.astype(np.int32)
  results['error'] = targets['charges'] - results['charges']

  st.markdown('{} records'.format(results.shape[0]))
  if verbose:
    st.info('results = {} {}'.format(results.shape, results.dtypes.values), icon=emoji.emojize(':speech_balloon:'))

  st.table(results.head(HEAD_SIZE))

  ########################

  st.markdown('---')

  st.markdown('#### {} Dependancy'.format(emoji.emojize(':x-ray:')))
  st.table({
    'module': ['python', 'numpy', 'pandas', 'matplotlib', 'streamlit', 'tensorflow'],
    'version': [
      sys.version.split(maxsplit=1)[0], np.__version__, pd.__version__, mpl.__version__, st.__version__,
      tf.__version__
    ]
  })

#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#
