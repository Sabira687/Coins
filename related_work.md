## 2. Related Work

### 2.1 Coin recognition

Recent deep-learning research on coins has concentrated largely on ancient coins, and Cirillo et al. [1] note that modern circulating coins have received little attention from AI research. They address this gap with a dataset of euro coin images showing the relief side and classify both the issuing country and the value with convolutional networks and self-attention layers. Among the difficulties they report are similar designs reused across coin versions and changes in relief over time. Kazakhstan tenge coins present a related difficulty: the 20 and 50 tenge coins have the same colour and nearly the same design. Like most classification studies, [1] measures how well models separate the classes; it does not examine what visual evidence the models rely on, which is the question addressed here.

### 2.2 Pretrained representations

Transfer learning from ImageNet is the usual starting point when labelled data are scarce. Kornblith et al. [2] compared a broad set of ImageNet architectures on twelve image classification datasets, using them as fixed feature extractors, fine-tuning them, and training them from random initialisation. After fine-tuning, transfer accuracy tracked ImageNet accuracy closely, yet on two small fine-grained datasets ImageNet pretraining gave little benefit over training from scratch. Recognising coin denominations is a fine-grained problem of this kind, so the value of supervised pretraining for it cannot be assumed.

Self-supervised learning offers a different source of features. DINOv2 [3] pretrains vision transformers without labels on a curated and diverse collection of 142 million images and distils a billion-parameter model into a series of smaller ones. Its features are intended to transfer across image distributions and tasks, often without fine-tuning. Whether such general-purpose features also help a model generalise from a few physical objects to unseen ones remains open; we therefore compare a DINOv2 model with a supervised ImageNet ResNet and with a network trained from scratch.

### 2.3 Shortcuts and leakage in evaluation

Geirhos et al. [5] describe many failures of deep networks as shortcut learning: decision rules that perform well on standard benchmarks but do not transfer to more demanding test conditions. Among their recommendations is testing models on data that differ systematically from the training distribution. A classifier trained on a single coin per denomination can reach high accuracy by recognising that particular coin. A test on the same coin under new lighting cannot expose this shortcut; a test on different coins of the same denomination can.

A related problem is near-duplicate images shared by the training and test sets. Barz and Denzler [6] found near-duplicates of 3.3% of the CIFAR-10 test images and 10% of the CIFAR-100 test images in the respective training sets; after replacing them, classification accuracy fell by 9–14% relative to the original results. Frames extracted from the same video are near-duplicates by construction, which is why our data are split by video and by coin and never by individual frame.

### 2.4 Controlled datasets

Barbu et al. [4] argue that large computer vision datasets lack the experimental controls that are standard in other sciences, which allows models to exploit trivial correlations. Their ObjectNet test set randomises object backgrounds, rotations and viewpoints, and object recognition models lose 40–45% of their performance on it compared with other benchmarks. We apply the same principle on a much smaller scale: each part of our dataset varies one factor, either the recording conditions or the coin specimen, while the other is held fixed.

### 2.5 Visual explanations

Grad-CAM [7] builds class-specific localisation maps from the gradients that flow into the last convolutional layer, and its authors use it to analyse failure modes and to reveal dataset bias. We use Grad-CAM to compare where models trained on one coin and on several coins look when they classify the same unseen coins, and complement it with nearest-neighbour retrieval in the DINOv2 embedding space.

Taken together, previous work has improved coin classifiers, shown that pretraining does not always help fine-grained tasks, and documented how shortcuts and leakage inflate reported accuracy. We are not aware of coin recognition studies that separate the effect of recording conditions from the effect of the physical specimen; the present work does this with a controlled two-part dataset.

---

## References

[1] S. Cirillo, G. Solimando, and L. Virgili, "A deep learning approach to classify country and value of modern coins," *Neural Computing and Applications*, vol. 36, no. 20, pp. 11759–11775, 2024, doi: 10.1007/s00521-023-09355-6.

[2] S. Kornblith, J. Shlens, and Q. V. Le, "Do better ImageNet models transfer better?," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 2661–2671, doi: 10.1109/CVPR.2019.00277.

[3] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, "DINOv2: Learning robust visual features without supervision," *Transactions on Machine Learning Research*, 2024. [Online]. Available: arXiv:2304.07193.

[4] A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D. Gutfreund, J. Tenenbaum, and B. Katz, "ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019.

[5] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, no. 11, pp. 665–673, 2020, doi: 10.1038/s42256-020-00257-z.

[6] B. Barz and J. Denzler, "Do we train on test data? Purging CIFAR of near-duplicates," *Journal of Imaging*, vol. 6, no. 6, art. 41, 2020, doi: 10.3390/jimaging6060041.

[7] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, doi: 10.1109/ICCV.2017.74.
