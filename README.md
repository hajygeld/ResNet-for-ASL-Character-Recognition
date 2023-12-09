<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/catiaspsilva/README-template">
    <img src="images/gators.jpg" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">ASL Recognition using Deep Convolutional Neural Network</h3>

  <!-- <p align="center">
    Instructions to run this project and make predictions!
    <br />
    <a href="https://github.com/catiaspsilva/README-template/blob/main/images/docs.txt"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#usage"></a>
    ·
    <a href="https://github.com/catiaspsilva/README-template/issues">Report Bug</a>
    ·
    <a href="https://github.com/catiaspsilva/README-template/issues">Request Feature</a>
  </p> -->
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

In this project, we have created a Resnet model from sratch using Pytorch and trained it on a dataset of ASL images. Since we used ResNet, we named our group "The Three ResNeteers", but it was unfortunately misspelled in the repository name. To train this model, the original dataset provided had 8443 images with 9 classes for the letters A to I. We have augmented the dataset by applying transformation techniques such as rotations, shear, height and width shift. The final training dataset had over 32772 images with corresponding labels. The augmentation script can be found in `data_augmentation.py`. We split the data into 80/20 train/validation and trained the model on the train set using the script `train.py` and made predictions on the validation data using `test.py`. Below you can find information on how to use our model to make predictions.

[Here goes the title with hyperlink](https://github.com/catiaspsilva/README-template)

<!-- You can include tables or images to summarize your results when and if appropriate. -->

<!-- GETTING STARTED -->

## Getting Started

To recreate the project locally, simply clone this repository in your local computer. If you would like to train the model and get your weights, you can download our training dataset from this link [augmented_data](https://drive.google.com/drive/folders/1iRBEGGaEIdTs205GTU24fM8xOyqM4CIg?usp=sharing).

### Dependencies

Here is a list of all dependencies:

dependencies:

If you use a Pytorch 2.1.0 kernel in Hipergator, then all requirements should already be satisfied.

- python=3.10.12
- numpy=1.23.5
- torch=2.1.0+cu118
- torchvision=0.16.0+cu118
- h5py=3.9.0

<!-- - Pytorch 3.10.12
  ```sh
  conda install pytorch torchvision -c pytorch
  ``` -->

<!-- ### Alternative: Export your Environment -->

<!-- Alternatively, you can export your Python working environment, push it to your project's repository and allow users to clone it locally. This way, anyone can install it and they will have all dependencies needed. Here is how you export a copy of your Python environment:

```sh
conda env export > requirements.yml
```

The user will be able to recreate it using:

```sh
conda env create -f requirements.yml
``` -->

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/UF-FundMachineLearning-Fall23/final-project-code-report-thethreemustekeers.git
   ```
2. Setup (and activate) your environment

```sh
conda env create -f requirements.yml
```

<!-- USAGE EXAMPLES -->

## Usage

The weights of the model are saved in `Best_Model` directory in this repository.

# Train

To train the model from scratch, download the `augmented_data.npy` and `augmented_labels.npy` from the link [augmented_data](https://drive.google.com/drive/folders/1iRBEGGaEIdTs205GTU24fM8xOyqM4CIg?usp=sharing). Place the downloaded files in the empty `data/` folder within the project directory. Then from the project directory, open the train.ipynb file and run the cell. Alternatively, from the terminal, you can run the script `train.py` by

```
python3 train.py
```

This would then generate weights and save them in the `Best_Model/` folder within the project directory.

# Easy Test

To make predictions on the easy test set, (any test set with only images of ASL characters from A to I), run `test_easy.py` by

```
python3 test_easy.py "{test_data.npy}" "{test_labels.npy}"
```

Alternatively, you can run the `test_easy.ipynb`. To do so, open this file and change the file paths for the test data and labels in the main(). Then simply run the cell.

# Hard Test

To make predictions on the hard test set, (any test set with images of ASL characters from A to I belonging to the 9 classes and an additional class with arbitrary images), run `test_hard.py` by

```
python3 test_hard.py "{test_data.npy}" "{test_labels.npy}"
```

Alternatively, you can run the `test_hard.ipynb`. To do so, open this file and change the file paths for the test data and labels in the main(). Then simply run the cell.

<!-- _For more examples, please refer to the [Documentation](https://example.com)_ -->

<!-- ROADMAP -->

## Roadmap

See the [open issues](https://github.com/catiaspsilva/README-template/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- Authors -->

## Authors

Ahmed Rageeb Ahsan - [twitter](https://twitter.com/your_username) - email@example.com

Hajymyrat Geldimuradov - [twitter](https://twitter.com/your_username) - email@example.com

Md Jabir Hossain - [twitter](https://twitter.com/your_username) - email@example.com

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

- [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)
- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [Loaders.css](https://connoratherton.com/loaders)
- [Slick Carousel](https://kenwheeler.github.io/slick)

## Thank you

<!-- If this is useful: [![Buy me a coffee](https://www.buymeacoffee.com/assets/img/guidelines/download-assets-sm-1.svg)](https://www.buymeacoffee.com/catiaspsilva) -->
