## Image Cassification Task - Mask, Gender and Age Estimation    
<br>

**코드 사용 방법**

    CLI로 ./run_train.sh 를 실행하면, 설정한 값들로 python train.py가 실행됩니다.


* train.py 를 실행시켜 모델을 학습합니다.  
    
* model.py 에 사용하고자 하는 모델 class를 정의합니다. 모델을 불러오는 환경이 다르기 때문에 error가 날 수 있습니다. 수정해서 불러와주세요. torchvision.models에서 pretrain 모델을 가져왔습니다.
