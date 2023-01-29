# Online Continual Learning
ASER Code Analysis

## 명령어 (run)
python general_main.py --data cifar100 --cl_type nc --agent ER --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3 
* data: CIFAR-100
* cl_type (nc/ni): new class
* agent: Experience Replay
* update: ASER
* retrieve: ASER
* mem_size: 5000
* aser_type: asvm (Use mean values of Adversarial SV and Cooperative SV)
* n_smp_cls: Maximum number of samples per class for random sampling
* k: Number of nearest neighbors (K) to perform ASER

## general_main.py
* argument 지정
* multiple_run -> experiment/run.py

## run.py
* 데이터셋 준비 (setup)
* 모델 불러오기: cifar100 (Reduced_ResNet18)
* 버퍼 초기화 및 (업데이트, 추출) 방법 불러오기 -> agenets/exp_replay.oy
* 학습 (버퍼 retrieve+update) -> agents/exp_replay.py 
* 평가


## exp_replay.py
* before_train: 현재 테스크 데이터 불러오기 (배치)
* 버퍼에서 사용할 샘플 추출
* 추출한 샘플 + 현재 테스크 데이터로 학습
* 버퍼 업데이트
* after_train: before_train에서 설정한 내용 수정 

## aser_retrieve.py
* 버퍼가 차지 않았을 때: 랜덤으로 샘플링
* 버퍼가 찼을 때: ASER 샘플링으로 추출
    * 현재 테스크 데이터를 evaluation으로 설정 
        * Candidate 샘플은 랜덤 샘플링
        * Evaluation 샘플에 대해 candidate 샘플 ASV값 계산 (1) -> utils/aser_utils.py
    * 버퍼에서 candidate 샘플을 제외한 샘플을 evaluation으로 설정
        * Evaluation 샘플에 대해 candidate 샘플 ASV값 계산 (2) -> utils/aser_utils.py
    * if asvm: mean(2) - mean(1)가 가장 높은 샘플 추출
    * if asv: max(2) - min(1)가 가장 높은 샘플 추출