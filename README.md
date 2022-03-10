# Boostcamp AI Tech 3기: NLP-05-외않되조

---

# Project: 마스크 이미지 분류

- Wrap-up Report
    
[이미지_분류_NLP_팀_리포트(05조).pdf](https://github.com/NLP05/mask-classification/files/8219677/_._NLP_._.05.pdf)   

## Members

| 이름 | Github Profile | 역할 |
| --- | --- | --- |
| 공통 |  | EDA |
| 강나경 | angieKang | 모델 구조 잡기, 모델 성능 시각화 |
| 김산 | nasmik419 | age regression 관련 모델 조사, data augmentation |
| 김현지 | TB2715 | wandb 도입, 실험 환경 설정, age regression/classification |
| 정민지 | minji2744 | 모델 학습, 하이퍼파라미터 튜닝, 앙상블 |
| 최지연 | jeeyeon51 | 적용가능한 다양한 모델 조사, 3-headed 모델 학습 |

## 문제 개요

COVID-19의 전파를 차단하기 위해 코와 입을 완전히 가려 올바르게 마스크를 착용해야 합니다.

넓은 공공장소에서 카메라로 비춰진 사람의 얼굴 이미지만으로 올바른 마스크 착용 유무를 자동으로 판단할 수 있다면 적은 인적자원으로도 충분히 마스크 착용 상태 검사가 가능할 것입니다.

이번 프로젝트에서는 5주간 딥러닝에 관해 배운 내용을 바탕으로 image classification을 진행하였습니다.

개발 환경은 GPU V100 (SSH connect)에서  Python으로 개발하였고, sub로 jupyter notebook를 사용하였습니다. 
협업 툴로는 notion과 github을 사용하였고 추가로 wandb와 slack을 통해 실험을 관리하였습니다.

## 모델 구조

- **3-Headed Model**
    
    ![model_structure](https://user-images.githubusercontent.com/59854630/157574345-897f0a24-5eff-4ff4-b0d6-f05a25815c1b.png)

    

- **Ensemble Model**
    
    ![Ensemble_Soft_voting](https://user-images.githubusercontent.com/59854630/157574257-a6caf7aa-bc2d-49a4-bdb7-198404537bec.jpg)
    

## 데이터셋 구조

- 마스크 착용여부, 성별, 나이를 기준으로 총 18개의 클래스가 있습니다.
    
    
    | Class | Mask | Gender | Age |
    | --- | --- | --- | --- |
    | 0 | Wear | Male | <30 |
    | 1 | Wear | Male | ≥30 and <60 |
    | 2 | Wear | Male | ≥60 |
    | 3 | Wear | Female | <30 |
    | 4 | Wear | Female | ≥30 and <60 |
    | 5 | Wear | Female | ≥60 |
    | 6 | Incorrect | Male | <30 |
    | 7 | Incorrect | Male | ≥30 and <60 |
    | 8 | Incorrect | Male | ≥60 |
    | 9 | Incorrect | Female | <30 |
    | 10 | Incorrect | Female | ≥30 and <60 |
    | 11 | Incorrect | Female | ≥60 |
    | 12 | Not Wear | Male | <30 |
    | 13 | Not Wear | Male | ≥30 and <60 |
    | 14 | Not Wear | Male | ≥60 |
    | 15 | Not Wear | Female | <30 |
    | 16 | Not Wear | Female | ≥30 and <60 |
    | 17 | Not Wear | Female | ≥60 |

## 실험 결과

### 리더보드 (대회 진행)

<img width="1086" alt="Untitled" src="https://user-images.githubusercontent.com/59854630/157574383-8987cf91-236d-4c85-8362-70a71d8edf8b.png">

- f1_score: 0.7459
- accuracy: 78.5714

### 리더보드 (최종)

<img width="1089" alt="Untitled 1" src="https://user-images.githubusercontent.com/59854630/157574366-1f97de46-a81c-4dcf-9e3c-8461ffa7e6c8.png">

- f1_score: 0.7301
- accuracy: 78.0794

## Requirements

> Confirmed that it runs on Ubuntu 18.04.5, Python  3.8, and pytorch 1.10.2
> 

필요한 패키지들은 아래 명령어를 입력하여 설치하실 수 있습니다. 

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install torchvision==0.11.3
conda install -c conda-forge tensorboard
conda install pandas=1.1.5
pip install opencv-python==4.5.1.48
conda install -c conda-forge scikit-learn=0.24.1
conda install -c conda-forge matplotlib=3.2.1
pip install python-dotenv
pip install wandb
```

## Getting Started

### 1. 코드 구조

- **dataset.py** : 데이터 라벨링 및 augmentation 기법 적용
- **evaluation.py** : inference를 통해 예측한 output.csv를 gt.csv와 비교하여 accuracy와 f1-score 계산
- **inference.py** : 학습된 모델로 예측 실행
- **loss.py** : loss function들을 정의
- **model.py** : efficientnet, resnet 등 다양한 모델들을 정의
- **train.py** : 파라미터들을 조절하여 학습 진행

### 2. 코드 실행 방법

1. 먼저 위의 [requirements](https://www.notion.so/README-md-8c42b02ed8f4435c9c588004b1e6d419) 참고해 환경설정을 진행합니다.
2. 모델 훈련
    1. `python3 train.py -epochs 10 --batch_size 32 --resize 300 300 --model EfficientNetB3 --optimizer AdamW --lr 0.0001 --name AdamW_gender_eb3 --augmentation CustomAugmentation --lr_decay_step 50 --log_interval 50 --criterion label_smoothing --lr_scheduler CosineAnnealing` 로 모델을 훈련하거나,
    2. `./run_train.sh` 명령어를 통해 `command_file.txt`에 정의해둔 하이퍼 파라미터 옵션을 수정한 모델들을 학습할 수 있습니다. 
        - command_file.txt: 진행하는 task 및 hyperparameter tuning 옵션을 세팅
3. `./run_inference.sh` 명령어로 `command_inference.txt`에 설정한 경로의 모델을 활용해 추론한 결과를 csv로 저장합니다.
