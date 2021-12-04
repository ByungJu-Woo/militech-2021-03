# Militech-2021-03
Reseach about adversarial patches suitable for military domains

## Research Info
 AI(Artificial Intelligence)기술의 발전은 객체 탐지와 감시기술을 통해 국방의 중요한 부분으로 꼽힌다. 적대적 공격(Adversarial Attack)은 급격하게 발달하는 딥러닝 모델을 이용한 컴퓨터 비전 기술이 고도화됨에 따라 이에 대항하여 급격하게 발전하고 있다. 그 중에서, ‘적대적 패치(Adversarial Patch)’를 활용한 방식은 일정한 크기의 패치를 생성하여 공격 대상 물체에 부착하기만 하면 되므로 실제 물리적 대상에 대해서 실시간(Real-time)으로 수월하게 공격을 수행할 수 있다는 장점이 있다. 기존 연구들은 실시간 탐지 모델 YOLOv2를 회피하는 적대적 패치를 제시하고 있지만, 군복에 부착시 위장성이 떨어진다는 한계점을 갖고 있었다. 이에 이전 적대적 패치와 비슷한 성능을 유지하면서도 국방 분야에 특화된 패치를 만들고자 한다. 본 연구에서는 (1) 원본 이미지를 국군 전투복 무늬로 설정하고 변형, (2) MSE, camouflage, VGG perception과 같은 손실함수로 위장성과 회피성을 조절, (3) 기존의 정사각형 패치에 비해 세로 비율을 2배 늘려 높은 회피성능을 유도하는 세 방법을 시도하여 디지털 전투복 패턴(Military)과 유사한 패치를 생성하였다. 또한 INRIA 데이터셋으로 학습한 패치의 성능을 검증하고, 해당 패치를 인쇄하여 물리적 환경에서 테스트함으로써 우리의 패치가 실시간으로 적대적 공격을 수행하는 모습을 확인함으로써 본 연구의 패치가 기존의 적대적 패치보다 효과적으로 시각적 위장을 제공한다는 사실을 확인하였다. 우리가 아는 한, 본 연구는 실제 전투복 무늬에 대하여 객체인식 인공지능에 대한 적대적 공격(Adversarial attack)과 시각적 위장(Visual camouflage)을 동시에 시도한 최초의 사례이다.

 
## Related Links
본 Repository는 EAVAISE의 Adversarial YOLO 구현을 기반으로 한다 : https://gitlab.com/EAVISE/adversarial-yolo   
본 Repository는 이 연구의 보고서를 따른다 : 보고서 링크  
본 연구의 Demo 영상은 다음에서 확인 가능하다 : https://www.youtube.com/playlist?list=PLuF6gVsuhzXLQ3RRzegTRQZsvZaoJJBk5
 
## How to Use
### Requirement
Python 3.6.버전을 사용한다. 또한 Pytorch가 설치되어 있어야한다 : https://pytorch.org/  
학습과정을 시각화하기 위해서, 아래와 같이 pip을 이용하여 tensorboardX를 설치 할 수 있다. 
```python
pip install tensorboardX tensorboard
```  
학습에 필요한 YOLOv2 MS COCO weights 및 INRIA dataset은 Repository에 모두 저장되어 있다.

---
### Training Patch
train_patch.py를 수정하여 적대적패치의 시작 이미지, 유사성을 대조할 이미지, Loss 등을 변경할 수 있다.  
아래의 커맨드를 이용하여 패치를 학습시킨다
```python
python train_patch.py paper_obj
```  
학습 과정에서 매 epoch 마다 패치는 pics 폴더에 저장된다  

---

### Testing Patch
아래의 커맨드를 이용하여 패치를 적용하여 성능을 확인한다. 실행시, result/test_military_detection 폴더에 패치를 적용한 INRIA dataset의 detection 결과가 저장된다  
```python
python test_military.py paper_obj patch_name
```    

아래의 커맨드를 이용하여 패치의 성능을 평가한다. 실행시, recall, precision, fscore 값을 mat 파일로 얻을 수 있다
```python
python eval_military.py paper_obj matfile_name
```    
 
---
### command.ipynb
우리는 google colab 환경에서 연구를 수행하였다.  
adversarial-yolo, custom_model_training 2개의 폴더를 구글 드라이브에 저장하고, command.ipynb의 메뉴얼을 따르면 별도의 환경설정 및 경로 변경 없이 학습 및 테스트를 수행할 수 있을 것이다.  

