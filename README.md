# Music-Description-Generator

## Example Outputs

#### Ex 1

https://github.com/ongdyub/Music-Description/assets/88565572/48438cbf-cd37-4156-af54-649c3b008db8

```python
'This is a classical music piece. It could also be playing in the background at a coffee shop.'
```

#### Ex 2

https://github.com/ongdyub/Music-Description/assets/88565572/3da542d2-a078-4620-9b0f-4239f0983c8e

```python
'The low quality recording features a live performance of a folk song and it consists of groovy bass, shimmering hi hats, soft kick and harmonizing vocals, harmonizing vocals. It sounds energetic.'
```

## Model Architecture

#### Audio Encoder

Use facebook/encodec_32khz huggingface pre-trained model.

Input is 10 seconds of raw audio, sample rate is 32000.

Audio Encoder convert raw audio to Discrete sequence of audio like [100, 321, 210, 124, ... , 213].

Sequence of audio codebook is input of Text Decoder.

#### Text Decoder

Use Transformer base architecture and T5 tokenizer.

More details (nLayers, hidden dim, nHeads, etc...) are in trainer.ipynb

Input is sequence of codebook index, Out is sentences.

<img src="./data/arch.png"/>

## Training & Test Loss Graph

<img src="./img/loss_graph.png"/>


