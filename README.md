![image](https://user-images.githubusercontent.com/82706646/200876491-56725e15-2ca2-412f-b78b-507e631d4cc9.png)

# Rec5der's Book Rating Predictions
- Naver Bosst Camp AI tech 4th
- Recsys 비공개대회


## 📚 Project Abstract
- 목적 : 사용자의 책 평점 데이터를 바탕으로 사용자가 어떤 책을 더 선호할지 예측하기 위함
- task : 26,167명의 사용자(user)가 52,000개의 책(item)에 대해 남길 것으로 기대하는 76,699건의 평점(rating) 예측
- 평가지표 : RMSE


## 🎥 Team
| Name | Github | Role |
| :-: | :-: | --- |
| 배성재 (T4097) | [link](https://github.com/SeongJaeBae) | 모델 실험, wandb |
| 양승훈 (T4122) | [link](https://github.com/Seunghoon-Schini-Yang) | 데이터 전처리 |
| 정민주 (T4192) | [link](https://github.com/jeongminju0815) | 모델 실험 |
| 조수연 (T4208) | [link](https://github.com/Suyeonnie) | 모델 실험 |
| 황선태 (T4236) | [link](https://github.com/HSUNEH) | rule based |


## 🔎 Experiments
- ML : Catbosstregressor, Gradientboostregressor
- DL : FM, FFM, NCF, WDN, DCN, CNN_FM, DeepCoNN
- Rule Based : 평균, 표준편차 활용


## ⭐️ Result
- Public RMSE : 2.1831
- Private RMSE : 2.1814
- ![image](https://user-images.githubusercontent.com/82706646/200873968-468c4e78-643d-4acf-9132-540ab4245838.png)
