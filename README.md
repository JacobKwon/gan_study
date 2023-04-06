# GAN Study
### 프로젝트 내용
- 2023.02.27-2023.04.06
- 빠르게 늘어나는 여러가지 생성 모델을 사용하고 결과값을 확인하기 위한 Side project

### 개발환경
- Apple Silicon M1 MacMini CTO, MacBook Pro M1, Personal Ubuntu Desktop server
- Python 3.8, Tensorflow-ROCm, OpenCV, Sklearn, Pandas ...

### 팀 구성
<table>
  <tr>
    <th>이름</th>
    <th>역할</th>
    <th>이메일</th>
  </tr>
  
  <tr>
    <th><a href='https://github.com/JacobKwon'>권진욱</a></th>
    <th>Leader</th>
    <th>apodis1991@gmail.com</th>
  </tr>
  
  <tr>
    <th><a href='https://github.com/worldpapa'>김세상</a></th>
    <th>Member</th>
    <th>sstptkdss1@icloud.com</th>
  </tr>
  
  <tr>
    <th><a href='https://github.com/devTeddyB'>전대광</a></th>
    <th>Member</th>
    <th>eorhkd2626@gmail.com</th>
  </tr>
  
  <tr>
    <th><a href='https://github.com/ggydo59'>민경도</a></th>
    <th>Member</th>
    <th>mgd813@gmail.com</th>
  </tr>
</table>

<br/><hr/>
<details>
  <summary>Reference</summary>
  <br/>
  
  [얼굴 인식 알고리즘 선행 연구를 소개합니다](https://tech.kakaoenterprise.com/63)<br/>
  [GAN 적대적 생성 신경망과 이미지 생성 및 변환 기술 동향](https://ettrends.etri.re.kr/ettrends/184/0905184009/35-4_91-102.pdf)<br/>
  [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)<br/>
  [A GAN-Based Face Rotation for Artistic Portraits](https://www.mdpi.com/2227-7390/10/20/3860)<br/>
  [TP-GAN](https://github.com/HRLTY/TP-GAN)<br/>
  [Pose-Guided Photorealistic Face Rotation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Pose-Guided_Photorealistic_Face_CVPR_2018_paper.pdf)<br/>
  [Pose Guided Person Image Generation](https://arxiv.org/abs/1705.09368)<br/>
  [FaceID-GAN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_FaceID-GAN_Learning_a_CVPR_2018_paper.pdf)<br/>
  [Cross-Domain Face Synthesis using a Controllable GAN](https://arxiv.org/abs/1910.14247)<br/>
  [Exposing GAN](https://arxiv.org/pdf/1904.00167.pdf)<br/>
  [LANDMARKGAN](https://arxiv.org/pdf/2011.00269v2.pdf)<br/>
  [Style-GAN](https://keras.io/examples/generative/stylegan/)<br/>
  [Nickface using Generative Adversarial Networks](https://tech.kakao.com/2021/08/04/nickface/)<br/>
  [CelebA Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)<br/>
  [CelebA Progressive GAN](https://www.tensorflow.org/hub/tutorials/tf_hub_generative_image_module?hl=ko)<br/>
  [StyleGAN Review](https://www.youtube.com/watch?v=HXgfw3Z5zRo)<br/>
  [DCGAN Review](https://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html)<br/>
  [awesome GAN applications demo](https://github.com/nashory/gans-awesome-applications)<br/>
  [MoFA](https://arxiv.org/pdf/1703.10580.pdf)<br/>
  [Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction](https://www.youtube.com/watch?v=uIMpHZYB8fI)<br/>
  [Face landmarks detection task guide](https://google.github.io/mediapipe/solutions/face_mesh)<br/>
  [Face Image Completion Based on GAN Prior](https://www.mdpi.com/2079-9292/11/13/1997)<br/>
  [Face Mesh Tutorial](https://morioh.com/p/23596f7e6d56)<br/>
  [Pre-Trained Feature Fusion and Multidomain Identification Generative Adversarial Network for Face Frontalization](https://ieeexplore.ieee.org/document/9837875)<br/>
  [Face Alignment](https://github.com/1adrianb/face-alignment)<br/>
  [tutorial_feature_homography](https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html)<br/>
  [tutorial_py_feature_homography](https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html)<br/>
  [GAN_LFW](https://github.com/bac423/GAN_LFW)<br/>
  [Pytorch LFW GAN-DCGAN](https://github.com/rootally/Generating-new-faces-using-Gans)<br/>
  [cv_note](https://s3.ap-northeast-2.amazonaws.com/doc.mindscale.kr/ngv/cv_note.pdf)<br/>
  [metric-learning-for-landmark-image-recognition](https://medium.com/mlearning-ai/metric-learning-for-landmark-image-recognition-6c1b8e0902bd)<br/>
  [Face2Face Note](https://towardsdatascience.com/face2face-a-pix2pix-demo-that-mimics-the-facial-expression-of-the-german-chancellor-b6771d65bf66)<br/>
  [Face2Face Demo](https://github.com/datitran/face2face-demo)<br/>
  [FaceConverter](https://github.com/taylorlu/FaceConverter)<br/>
  [PRNet](https://github.com/yfeng95/PRNet)<br/>
  [stargan-v2](https://github.com/clovaai/stargan-v2)<br/>
  [stargan paper](https://arxiv.org/abs/1912.01865)<br/>
  
  
</details>

<details>
  <summary>Datasets</summary>
  <br/>
  
  [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)<br/>
  [NVlabs ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)<br/>
  [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)<br/>
  [Open Images Dataset](https://storage.googleapis.com/openimages/web/visualizer/index.html)<br/>
  [Unsplash](https://unsplash.com/ko)<br/>
  [Pexels](https://www.pexels.com/ko-kr/)<br/>
</details>
