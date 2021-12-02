# militech-2021-03
Reseach about adversarial patches suitable for military domains

## Research Info
 --> 여기다 우리 보고서 초록 넣기 / 아니면 이 문단 아예 삭제하기
 
## Repository Info
본 Repository는 EAVAISE의 Adversarial YOLO 구현을 기반으로 한다 : https://gitlab.com/EAVISE/adversarial-yolo   
본 Repository는 이 연구의 보고서를 따른다 : 보고서 링크
 
## How to Use
### Requirement
Python 3.6.버전을 사용한다. 또한 Pytorch가 설치되어 있어야한다 : https://pytorch.org/  
학습과정을 시각화하기 위해서, 아래와 같이 pip을 이용하여 tensorboardX를 설치 할 수 있다. 
```python
pip install tensorboardX tensorboard
```  
학습에 필요한 YOLOv2 MS COCO weights 및 INRIA dataset은 Repository에 모두 저장되어 있다.

### Training Patch
train_patch.py를 수정하여 적대적패치의 시작 이미지, 유사성을 대조할 이미지, Loss 등을 변경할 수 있다.  
아래의 커맨드를 이용하여 패치를 학습시킨다
```python
python train_patch.py 
```  
학습 과정에서 매 epoch 마다 패치는 pics 폴더에 저장된다  


### Testing Patch
아래의 커맨드를 이용하여 패치를 적용하여 성능을 확인한다. 실행시, result/test_military_detection 폴더에 패치를 적용한 INRIA dataset의 detection 결과가 저장된다  
```python
python test_military.py patch_name
```    
---
아래의 커맨드를 이용하여 패치의 성능을 평가한다. 실행시, recall, precision, fscore 값을 mat 파일로 얻을 수 있다
```python
python eval_military.py matfile_name
```    
 
### command.ipynb
우리는 google colab 환경에서 연구를 수행하였다.  
adversarial-yolo, custom_model_training 2개의 폴더를 구글 드라이브에 저장하고, command.ipynb의 메뉴얼을 따르면 별도의 환경설정 및 경로 변경 없이 학습 및 테스트를 수행할 수 있을 것이다.  

