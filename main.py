import argparse
import time

import pandas as pd
import yaml

import wandb
from src import (CNN_FM, DeepCoNN, DeepCrossNetworkModel,
                 FactorizationMachineModel,
                 FieldAwareFactorizationMachineModel,
                 NeuralCollaborativeFiltering, WideAndDeepModel,
                 seed_everything)
from src.data import (context_data_load, context_data_loader,
                      context_data_split, dl_data_load, dl_data_loader,
                      dl_data_split, image_data_load, image_data_loader,
                      image_data_split, text_data_load, text_data_loader,
                      text_data_split)


def main(args):
    
    wandb.init(
        project="minju-test", 
        entity="boostcamp_l1_recsys05",
        name=f"experiment_{args.MODEL}", 
        # Track hyperparameters and run metadata
        config={
            "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE,
            "lr": args.LR
            })
    seed_everything(args.SEED)

    ############## WANDB START
    wandb.init(
        project="schini-test", 
        entity="boostcamp_l1_recsys05",
        name=f"experiment_{args.MODEL}", 
        # Track hyperparameters and run metadata
        config={
            "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE,
            "lr": args.LR,
            "embed_dim": 16
            })
    
    """
    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'rmse'}, 
        'parameters':{
            'batch_size': {'max': 2048, 'min': 512},
            'epochs': {'max': 20, 'min': 5},
            'lr': {'max': 0.002, 'min': 0.0005 }
            }}   
    sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'rmse'}, 
    'parameters':{
        'emb_dim':{'max': 32, 'min': 8  }
        }}
    config={
            "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE,
            "lr": args.LR,
            "emb_dim": 16
<<<<<<< HEAD
            }
<<<<<<< HEAD
    # print(args.NCF_MLP_DIMS)
    # print(type(args.NCF_MLP_DIMS))
    # print(args.NCF_MLP_DIMS[0])
    # tmp = (2,2)
    # print(tmp)
    # print(type(tmp))
    # input()
=======
    
>>>>>>> cb089bf3c2ef8c46bc78d583bd05fd2c824a5d7b
=======
            }"""
    
>>>>>>> 7adb1188234bd2d679a59e645a4d59d61dd6637c
    ######################## DATA LOAD
    print(f'--------------- {args.MODEL} Load Data ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_load(args)
    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.MODEL == 'CNN_FM':
        data = image_data_load(args)
    elif args.MODEL == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    else:
        pass

    ######################## Train/Valid Split
    print(f'--------------- {args.MODEL} Train/Valid Split ---------------')
    if args.MODEL in ('FM', 'FFM'):
        data = context_data_split(args, data)
        train_dataset, valid_dataset, data = context_data_loader(args, data)

    elif args.MODEL in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.MODEL=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.MODEL=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    else:
        pass
    




    ######################## Model
    print(f'--------------- INIT {args.MODEL} ---------------')
    if args.MODEL=='FM':
        model = FactorizationMachineModel(args, train_dataset, valid_dataset, data)
    elif args.MODEL=='FFM':
        model = FieldAwareFactorizationMachineModel(args, data)
    elif args.MODEL=='NCF':
        model = NeuralCollaborativeFiltering(args, data)
    elif args.MODEL=='WDN':
        model = WideAndDeepModel(args, data)
    elif args.MODEL=='DCN':
        model = DeepCrossNetworkModel(args, data)
    elif args.MODEL=='CNN_FM':
        model = CNN_FM(args, data)
    elif args.MODEL=='DeepCoNN':
        model = DeepCoNN(args, data)
    else:
        pass

<<<<<<< HEAD
    # wandb.config.update(args)
=======
    #wandb.config.update(args)
>>>>>>> 7adb1188234bd2d679a59e645a4d59d61dd6637c
    # wandb.watch(model)

    ######################## TRAIN
    print(f'--------------- {args.MODEL} TRAINING ---------------')
    rmse = model.train()
    
    ######################## INFERENCE
    print(f'--------------- {args.MODEL} PREDICT ---------------')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN'):
        predicts = model.predict(data['test_dataloader'])
    elif args.MODEL=='CNN_FM':
        predicts  = model.predict(data['test_dataloader'])
    elif args.MODEL=='DeepCoNN':
        predicts  = model.predict(data['test_dataloader'])
    else:
        pass
    
    ######################## PREDICT GET IN RANGE
    def adjust_predict(y):
        if y < 1.0:
            return 1.0
        elif y > 10.0:
            return 10.0
        return y
    
    predicts = list(map(adjust_predict, predicts))

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.MODEL} PREDICT ---------------')
    submission = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')
    if args.MODEL in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'):
        submission['rating'] = predicts
    else:
        pass
    
    

    #기존 파일 저장 방식
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    submission.to_csv('submit/{}_{}_{}_{}.csv'.format(save_time, args.MODEL, round(rmse, 5), 'origin'), index=False)

    
    # '''
    # rule-based
    # '''
    # #########################
    # train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    # n_thres = 5; k = 0.5
    # #########################

    # # user-based
    # count = train.groupby("user_id").size()
    # dfcount = pd.DataFrame(count, columns=["count"])
    # train = pd.merge(train, dfcount, how='left', on='user_id')
    # submission = pd.merge(submission, dfcount, how='left', on='user_id')
    # submission['count'] = submission['count'].fillna(0)
    # submission.set_index("user_id", inplace = True)

    # for row in submission.itertuples():
    #     if row[3] == 0 :                                       # train에서 등장하지 않았던 user_id
    #         submission.at[row[0],'rating'] = 7                 # 1개 user_id 평균
    #     else:
    #         if train[train['user_id']==row[0]]['count'].mean() >= n_thres :       # n_thres 번 이상 rating 매긴 user_id
    #             if train[train['user_id']==row[0]]['rating'].std() <= k:          # 각자 매긴 rating 의 표준편차가 k 이하인 사람에게만 적용
    #                 # submission.at[row[0], 'rating'] = train[train['isbn']==row[0]]['rating'].mean()
    #                 ##### 표준편차 기준 더 작게 하고 round 씌워서 돌려보기
    #                 submission.at[row[0], 'rating'] = round(train[train['user_id']==row[0]]['rating'].mean())
    #                 #####
    

    # # drop "count"
    # submission = submission.drop(['count'], axis=1)
    # train = train.drop(['count'], axis=1)
    # submission = submission.reset_index()

    # # item-based
    # count = train.groupby('isbn').size()
    # dfcount = pd.DataFrame(count, columns=['count'])
    # train = pd.merge(train, dfcount, how='left', on='isbn')
    # submission = pd.merge(submission, dfcount, how='left', on='isbn')
    # submission['count'] = submission['count'].fillna(0)
    # isbnlist = set()
    # for row in submission.itertuples():
    #     if row[0] not in isbnlist:
    #         if row[3] == 0 :                                                   # train에서 등장하지 않았던 isbn
    #             submission.at[row[0], 'rating'] = 6.884027966331795            # 1개 isbn 평균
    #         else:
    #             if train[train['isbn']==row[1]]['count'].mean() >= n_thres :   # n_thres 명 이상 rating 매긴 isbn
    #                 if train[train['user_id']== row[1]]['rating'].std() <= k:  # 각자 매긴 rating 의 표준편차가 k 이하인 사람에게만 적용
    #                     # submission.at[row[0],'rating'] = train[train['isbn']== row[0]]['rating'].mean()
    #                     ##### 표준편차 기준 더 작게 하고 round 씌워서 돌려보기
    #                     submission.at[row[0], 'rating'] = round(train[train['isbn']==row[1]]['rating'].mean())
    #                     #####
    #         isbnlist.add(row[0])
    # submission = submission.reset_index()
    # submission = submission.drop(['count'], axis=1)


    # now = time.localtime()
    # now_date = time.strftime('%Y%m%d', now)
    # now_hour = time.strftime('%X', now)
    # save_time = now_date + '_' + now_hour.replace(':', '')
    # submission.to_csv('submit/{}_{}_{}_{}_{}_{}.csv'.format(save_time, args.MODEL, round(rmse, 5), n_thres, k, 'user_isbn', index=False))
    # #########################
    

if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--DATA_PATH', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--MODEL', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--DATA_SHUFFLE', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--TEST_SIZE', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--SEED', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    
    ############### TRAINING OPTION
    arg('--BATCH_SIZE', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--EPOCHS', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--LR', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--WEIGHT_DECAY', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')
    arg('--SPLIT_OPT', type=str, default='tts', help='train-test-split 옵션을 선택할 수 있습니다. (tts / kfold / skf)')

    ############### GPU
    arg('--DEVICE', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')

    ############### FM
    arg('--FM_EMBED_DIM', type=int, default=16, help='FM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### FFM
    arg('--FFM_EMBED_DIM', type=int, default=16, help='FFM에서 embedding시킬 차원을 조정할 수 있습니다.')

    ############### NCF
    arg('--NCF_EMBED_DIM', type=int, default=16, help='NCF에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--NCF_MLP_DIMS', type=list, default=(16, 16), help='NCF에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--NCF_DROPOUT', type=float, default=0.2, help='NCF에서 Dropout rate를 조정할 수 있습니다.')

    ############### WDN
    arg('--WDN_EMBED_DIM', type=int, default=16, help='WDN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--WDN_MLP_DIMS', type=list, default=(16, 16), help='WDN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--WDN_DROPOUT', type=float, default=0.2, help='WDN에서 Dropout rate를 조정할 수 있습니다.')

    ############### DCN
    arg('--DCN_EMBED_DIM', type=int, default=16, help='DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DCN_MLP_DIMS', type=list, default=(16, 16), help='DCN에서 MLP Network의 차원을 조정할 수 있습니다.')
    arg('--DCN_DROPOUT', type=float, default=0.2, help='DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--DCN_NUM_LAYERS', type=int, default=3, help='DCN에서 Cross Network의 레이어 수를 조정할 수 있습니다.')

    ############### CNN_FM
    arg('--CNN_FM_EMBED_DIM', type=int, default=128, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--CNN_FM_LATENT_DIM', type=int, default=8, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')

    ############### DeepCoNN
    arg('--DEEPCONN_VECTOR_CREATE', type=bool, default=False, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--DEEPCONN_EMBED_DIM', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--DEEPCONN_LATENT_DIM', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--DEEPCONN_CONV_1D_OUT_DIM', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_KERNEL_SIZE', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_WORD_DIM', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--DEEPCONN_OUT_DIM', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    
    ############### WANDB
    arg('--WANDB_SWEEP', type=bool, default=False, help='WANDB_SWEEP을 돌렸을 떄 TRUE 로 설정')
    
    args = parser.parse_args()
    main(args)
