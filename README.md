# SuperCodec: A Neural Speech Codec with Selective Back-Projection Network
[![githubio](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat)](https://exercise-book-yq.github.io/SuperCodec-Demo/)

## Updates

- Code release. (Jul. 27, 2024)
- Online demo at Github See [here](https://exercise-book-yq.github.io/SuperCodec-Demo/). (Aug. 13, 2023)
- Supports 16-48 kHz at variable bitrates. (Jul. 27, 2024)

In this [paper](https://arxiv.org/abs/2407.20530), we present SuperCodec, a neural speech codec that replaces the standard feedforward up- and downsampling layers with Selective Up-sampling Back Projection (SUBP) and Selective Down-sampling Back Projection (SDBP) modules. Our proposed method efficiently preserves the information, on the one hand, and attains rich features from lower to higher layers of the network, on the other. Additionally, we propose a selective feature fusion block in the SUBP and SDBP to consolidate the input feature maps

<table style="width:100%">
  <tr>
    <td><img src="./resources/supercodec.png" alt="inference" height="240"></td>
  </tr>
  <tr>
    <th>Supercodec</th>
  </tr>
</table>

## Pre-requisites

1. Clone this repo: `git clone https://github.com/exercise-book-yq/Supercodec.git`

2. CD into this repo: `cd Supercodec`

3. Install python requirements: `pip install -r requirements.txt`


## Training Example

```python
# train
python train.py --config config_v1.json
```

## Inference Example

```python
# inference
python inferece.py --checkpoint_file [generator checkpoint file path]
```

## References

- https://github.com/jik876/hifi-gan
- https://github.com/OlaWod/FreeVC/tree/main
- https://github.com/lucidrains/audiolm-pytorch
