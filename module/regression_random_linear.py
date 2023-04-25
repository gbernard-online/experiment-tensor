#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#

import sys

import emoji

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st

import tensorflow as tf

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

  width = st.sidebar.select_slider('width', [100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000], 500)
  offset = st.sidebar.slider('offset', -5.0, 5.0, 0.0, step=0.5)

  st.sidebar.markdown('---')

  noise = st.sidebar.slider('noise', 0.0, 0.5, 0.1, step=0.05)

  st.sidebar.markdown('---')

  reign = st.sidebar.select_slider('reign', range(5, 55, 5), 10)

  st.sidebar.markdown('---')

  ########################

  st.markdown('---')
  st.markdown('#### {} Dataset'.format(emoji.emojize(':jar:')))

  st.markdown('{} samples'.format(width))

  features = tf.random.normal((width, 1)) + offset

  if noise:
    targets = features + tf.random.normal(features.shape, 0, noise)
  else:
    targets = features

  if verbose:
    st.info('features = {} [{}]'.format(features.shape, features.dtype), icon=emoji.emojize(':speech_balloon:'))
    st.info('targets = {} [{}]'.format(targets.shape, features.dtype), icon=emoji.emojize(':speech_balloon:'))

  ########################

  columns = st.columns([1, 1])

  figure, axe = plt.subplots(figsize=(10, 10))

  axe.scatter(features, targets, alpha=0.5)
  columns[0].pyplot(figure)

  plt.close(figure)
  del figure, axe

  del columns

  ########################

  st.markdown('---')
  st.markdown('#### {} Model'.format(emoji.emojize(':abacus:')))

  st.markdown('{} epoches'.format(reign))

  model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, name='output')])

  if verbose:
    st.info('model = {} [{}]'.format(model, model.dtype), icon=emoji.emojize(':speech_balloon:'))

  model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=tf.keras.optimizers.SGD())

  ########################

  frame = st.empty()

  columns = st.columns(5)

  losses = []
  for number in range(reign):

    history = model.fit(features, targets, batch_size=10, verbose=0).history
    losses.append(round(history['loss'][0], 4))

    figure, axe = plt.subplots(figsize=(20, 6))
    axe.set_xlabel('epoch')
    axe.set_ylabel('loss')
    axe.set_xlim([-0.1, reign - 0.9])
    axe.set_xticks(range(0, reign))

    axe.plot(range(0, len(losses)), losses, '-o', c=palette[2], linewidth=3.0)
    frame.pyplot(figure)

    plt.close(figure)
    del figure, axe

    targets_predicted = model.predict(features, verbose=0)

    with columns[number % len(columns)]:

      figure, axe = plt.subplots(figsize=(5, 5))
      axe.set_title('epoch={}'.format(number))

      axe.scatter(features, targets_predicted, alpha=0.5)
      st.pyplot(figure)

      plt.close(figure)
      del figure, axe

  del columns

  ########################

  st.markdown('loss = {:0.4f}'.format(losses[-1]))

  with st.expander(emoji.emojize(':information:')):
    st.table({'loss': losses})

  ########################

  st.markdown('---')

  st.markdown('#### {} Dependancy'.format(emoji.emojize(':x-ray:')))
  st.table({
    'module': ['python', 'matplotlib', 'streamlit', 'tensorflow'],
    'version': [sys.version.split(maxsplit=1)[0], mpl.__version__, st.__version__, tf.__version__]
  })

#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#
