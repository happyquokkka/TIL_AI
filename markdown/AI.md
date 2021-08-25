# AI

#### Artificial Intelligence : 인공지능

* System that think humans
  * 인지과학(cognitive science)적 접근 : 인간의 사고과정을 모방
    * 생각을 하려면 먼저 알아야 한다 (= 선先학습)
* System that act like humans
  * Turing test : 인간의 행동과 컴퓨터의 행동을 구분하지 못하도록 
    * ex) "로봇이 아닙니까?" (다음 그림 중 나무가 포함된 그림을 모두 선택하시오)



#### 종류

* AI : 컴퓨터가 인간의 사고를 모방 / 사람과 컴퓨터를 구분하지 못함 (사람같은 컴퓨터)
  * ML : 컴퓨터가 스스로 학습 / 함수 기반이 아닌 데이터 기반
    * DL : Perceptron (인간의 뉴런을 참조)



* Machine Learning : 입력 →  학습 →  결과
* Deep Learning : 다층 신경망(Neural Network)을 사용해 학습
* Reinforcement Learning : 행동(action)을 통해 상태(state)를 변경하여 보상(reward)을 획득



* 지도학습 (Supervised Learning) : 문제와 답을 같이 제공
  * 예측 (prediction - linear regression)
  * 분류 (classification - logistic regression = multiple linear regression)
* 비지도학습 (UnSupervised Learning)
  * 군집 (clustering) : 제공한 데이터가 어떤 집단에 속하는지 판단
* 강화학습 (Reinforcement Learning)
  * 보상 (reward base) : 효율적으로 보상을 획득하는 방법





### ML

#### 개념

* 기존 방식
  * input (x) → function (x) → output (y)
* 기계 학습
  * training data(x, y) + learning = Model (Hypothesis)
  * test data(x) → Model → output(y)
* h(y) = W * X + b
  * 학습 : Weight, bias 를 변경하는 일련의 과정
* 순서
  * 데이터 준비
    * 학습에 필요한 데이터 준비단계 (전처리)
    * 결측치(Null) 와 이상치(정상 범주에서 심하게 벗어나는 값)를 제거하거나 최빈값, 평균값, 중위값으로 대치
  * 데이터 분할 
    * train set 과 test set 으로 분리 (보통 7:3 의 비율로 분리함)
      * __training data set__ : 모델 학습용
      * __validation set__ : 모델 검증용 → 여러 번 평가
        * ex) K겹 교차검증
      * __test data set__ : 모델 성능 평가용 → 단 한 번 평가
  * 모델 준비
    * 사용할 모델 결정
    * Prediction (예측모델) - Linear regression 
    * Classification (분류모델) - Logistic regression (=Multiple linear regression)
    * Clustering (군집)
  * 학습
    * 데이터를 가지고 모델 학습
      * activation function : 활성화 함수
      * loss function (cost function) : h와 y(실제 결과값)의 차이 = 오차
        * 오차가 적다 = 많이 맞췄다!
        * 오차 제곱법 : 오차를 제곱한 2차함수 그래프의 기울기가 0인 점을 찾기
      * Optimization : 최적화
      * Hyperparameter Tuning : 가장 효율적인 값 선택
  * 예측 및 평가
    * test data set을 가지고 학습이 제대로 되었는지 검증
    * 정확도 (accuracy) 
    * 정밀도 (precision), 재현율 (recall)



#### 종류

1. 예측 (prediction)

   : 연속적인 데이터 → 수치 계산 / linear regression

   * 선형회귀분석
   * 데이터들을 가장 잘 표현할 수 있는 선을 생성 → 입력에 대한 예측

2. 분류 (classification)

   : 불연속적인 데이터 → 확률 계산 / logistic regression ( = multiple linear regression)

   * 데이터들을 분류하자
   * binary classification : true / false
   * multi-label classification : a, b, c, d, ...
     * one hot encoding : 모든 레이블을 합치면 1이 됨

3. 





### RL

* 긍정적인 행동 → 보상 → 행동의 반복 (상호작용) → 행동 교정
* _Reward base_
* computational approach to learning from interaction
* agent (학습시킬 주체) 가 특정 action을 취했을 때 그 action에 의해 state를 가지게 됨
  * 그 state 는 environment 에 정의되어 있음
  * reward 가 1이 되면 프로그램은 종료됨

