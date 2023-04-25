#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#

import sys

import emoji

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st

import sklearn
import sklearn.datasets

import tensorflow as tf

#----------------------------------------------------------------------------------------------------------------------#

STEP_SIZE = 50

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

  st.sidebar.markdown('### {} parameters'.format(emoji.emojize(':gear:')))

  width = st.sidebar.select_slider('width', [500, 1000, 2000, 5000], 1000)

  st.sidebar.markdown('---')

  noise = st.sidebar.slider('noise', 0.00, 0.10, 0.03, step=0.01)

  st.sidebar.markdown('---')

  height = st.sidebar.select_slider('height', [2**number for number in range(2, 6)], 8)
  reign = st.sidebar.select_slider('reign', range(5, 65, 5), 15)

  st.sidebar.markdown('---')

  ########################

  st.markdown('---')
  st.markdown('#### {} Dataset'.format(emoji.emojize(':jar:')))

  st.markdown('{} samples'.format(width))

  features, targets = sklearn.datasets.make_circles(n_samples=width, noise=noise)

  if verbose:
    st.info('features = {} [{}]'.format(features.shape, features.dtype), icon=emoji.emojize(':speech_balloon:'))
    st.info('targets = {} [{}]'.format(targets.shape, targets.dtype), icon=emoji.emojize(':speech_balloon:'))

  ########################

  columns = st.columns([1, 1])

  figure, axe = plt.subplots(figsize=(10, 10))

  axe.scatter(features[:, 0], features[:, 1], alpha=0.5, c=targets)
  columns[0].pyplot(figure)

  plt.close(figure)
  del figure, axe

  del columns

  ########################

  st.markdown('---')
  st.markdown('#### {} Model'.format(emoji.emojize(':abacus:')))

  st.markdown('{} epoches'.format(reign))

  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(height, activation='relu', name='intput'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
  ])

  if verbose:
    st.info('model = {} [{}]'.format(model, model.dtype), icon=emoji.emojize(':speech_balloon:'))

  model.compile(loss=tf.keras.losses.binary_crossentropy,
                metrics=[tf.keras.metrics.BinaryAccuracy()],
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

  ########################

  frame = st.empty()

  columns = st.columns(5)

  accuracies = []
  for number in range(reign):

    history = model.fit(features, targets, batch_size=20, verbose=0).history
    accuracies.append(round(history['binary_accuracy'][0], 4))

    figure, axe = plt.subplots(figsize=(20, 6))
    axe.set_xlabel('epoch')
    axe.set_ylabel('accuracy')
    axe.set_xlim([-0.1, reign - 0.9])
    axe.set_ylim([-0.02, 1.02])
    axe.set_xticks(range(0, reign))

    axe.plot(range(0, len(accuracies)), accuracies, '-o', c=palette[2], linewidth=3.0)
    frame.pyplot(figure)

    plt.close(figure)
    del figure, axe

    with columns[number % len(columns)]:

      features_predicted = np.round(model.predict(features, verbose=0))

      figure, axe = plt.subplots(figsize=(5, 5))
      axe.set_title('epoch={}'.format(number))

      axe.scatter(features[:, 0], features[:, 1], alpha=0.5, c=features_predicted)
      st.pyplot(figure)

      plt.close(figure)
      del figure, axe

      ########################

      meshes = np.meshgrid(np.linspace(features[:, 0].min(), features[:, 0].max(), num=STEP_SIZE),
                           np.linspace(features[:, 1].min(), features[:, 1].max(), num=STEP_SIZE))

      meshes_predicted = np.round(model.predict(np.c_[meshes[0].ravel(), meshes[1].ravel()], verbose=0))

      figure, axe = plt.subplots(figsize=(5, 5))
      axe.set_title('epoch={}'.format(number))

      axe.scatter(np.c_[meshes[0].ravel()], np.c_[meshes[1].ravel()], c=meshes_predicted, marker='s')
      st.pyplot(figure)

      plt.close(figure)
      del figure, axe

  del columns

  ########################

  st.markdown('accuracy = {:0.4f}'.format(accuracies[-1]))

  with st.expander(emoji.emojize(':information:')):
    st.table({'accuracy': accuracies})

  ########################

  st.markdown('---')

  st.markdown('#### {} Dependancy'.format(emoji.emojize(':x-ray:')))
  st.table({
    'module': ['python', 'numpy', 'matplotlib', 'streamlit', 'sklearn', 'tensorflow'],
    'version': [
      sys.version.split(maxsplit=1)[0], np.__version__, mpl.__version__, st.__version__, sklearn.__version__,
      tf.__version__
    ]
  })

#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#
