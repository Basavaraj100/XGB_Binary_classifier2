

def xgb_binary_classifier(train=None,
                        tests=[],
                        test_file_names=[],
                        target=None,
                        model_name='model',
                        index_col=None,
                        parameter_optimization=True,
                        show_n_report=5,
                        show_feature_importance=True,
                        param_grid={  
                                    "learning_rate":[.01,.015,.02,.025,.03,.035,.04,.045,.05],
                                    "gamma":[i/10.0 for i in range(0,5)],
                                    "max_depth": [2,3,4,5,6,7,8],
                                    "min_child_weight":[1,2,5,10],
                                    "max_delta_step":[0,1,2,5,10],
                                    "subsample":[i/10.0 for i in range(5,10)],
                                    "colsample_bytree":[i/10.0 for i in range(5,10)],
                                    "colsample_bylevel":[i/10.0 for i in range(5,10)],
                                    "reg_lambda":[1e-5, 1e-2, 0.1, 1, 100], 
                                    "reg_alpha":[1e-5, 1e-2, 0.1, 1, 100],
                                    "scale_pos_weight":[1,2,3,4,5,6,7,8,9,10,20,30,40],
                                    "n_estimators":[100,500,700,1000]},
                        n_iter=10,
                        CV=5,
                        scoring='roc_auc',
                        best_model=True,
                        learning_rate=0.1,
                        gamma=0,
                        max_depth=3,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=1,
                        colsample_bytree=1,
                        colsample_bylevel=1,
                        reg_lambda=1,
                        reg_alpha=0,
                        scale_pos_weight=1,
                        n_estimators=100,
                        objective='binary:logistic'
                                     ):


    '''
    This function build xgb binary classifier model and create reports
    '''


    # ====================================== defining required functions =====================
    def get_logger(name):
        import logging
        log_format = f"%(asctime)s  %(name)8s  %(levelname)5s  %(message)s at line no %(lineno)d"
        logging.basicConfig(level=logging.DEBUG,
                            format=log_format,
                            filename='log.log',
                            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(log_format))    
        logging.getLogger(name).addHandler(console)
        return logging.getLogger(name)
    # ----------------------------------gain function ----------------------------------
    
    def createGains(test,dep_var):
        import matplotlib.pyplot as plt
        import seaborn as sns 
        import warnings
        """
    createGains - A function to create Gains table and plot gains in Python
    =======================================================================================

    **createGains** is a Python function by MathMarket (Team Tesla) of TheMathCompany,
    It aims to create Gains for the model built dataset

    This function returns Gains table at index 0 and plot at index 1

    The module is built using matplotlib,pandas.
    Therefore, these libraries are need to be installed in order to use the module.

    The module consist of one function:
    `createGains(test dataset, dependent variable)`

    """
       
        try:
            warnings.filterwarnings('ignore')
        except Exception as e:
            print(e)
        try:
            test = test.sort_values(by='Pred_Prob',ascending=False)
        except Exception as e:
            print(e)
            print("Could not sort according to predicted probability.")
        try:
            test['bucket'] = pd.qcut(test.Pred_Prob, 10,duplicates='drop')
        except Exception as e:
            print(e)
            print("Could not create buckets of ten.")
        try:
            test['rank'] = test['Pred_Prob'].rank(method='first')
        except Exception as e:
            print(e)
        try:
            test['bucket'] = pd.qcut(test['rank'].values, 10).codes
        except Exception as e:
            print(e)
        try:
            test["DepVar"]=test[dep_var]
        except Exception as e:
            print(e)
        try:
            test["Non_Event"]=np.where(test["DepVar"] == 0 , 1,0)
        except Exception as e:
            print(e)
        try:
            grouped = test.groupby('bucket', as_index = True)
        except Exception as e:
            print(e)
        try:
            agg1 = pd.DataFrame()
        except Exception as e:
            print(e)
        try:
            agg1['min_score']=grouped.min().Pred_Prob
        except Exception as e:
            print(e)
        try:    
            agg1['max_scr'] = grouped.max().Pred_Prob
        except Exception as e:
            print(e)    
        try:
            agg1['DepVar'] = grouped.sum().DepVar
        except Exception as e:
            print(e)    
        try:
            agg1['Non_Event'] = grouped.sum().Non_Event
        except Exception as e:
            print(e)    
        try:    
            agg1['total'] = agg1.DepVar + agg1.Non_Event
        except Exception as e:
            print(e)    
        try:    
            agg2=agg1.iloc[: : -1]
        except Exception as e:
            print(e)    
        try:    
            pd.options.mode.chained_assignment = None
        except Exception as e:
            print(e)
        try:
            agg2.is_copy
        except Exception as e:
            print(e)
        try:
            agg2['Cumulative_Sum']=agg2['DepVar'].cumsum()
        except Exception as e:
            print(e)
        try:
            agg2['Perc_of_Events']= (agg2.DepVar/agg2.DepVar.sum()) * 100
        except Exception as e:
            print(e)
        try:
            agg2['Gain']=agg2['Perc_of_Events'].cumsum()
        except Exception as e:
            print(e)
        try:
            agg2.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(e)
        try:
            agg2["Bucket"]=[1,2,3,4,5,6,7,8,9,10]
        except Exception as e:
            print(e)
        try:
            agg2.min_score = agg2.min_score.round(3)
        except Exception as e:
            print(e)
        try:
            agg2.max_scr = agg2.max_scr.round(3)
        except Exception as e:
            print(e)
        try:
            agg2.Perc_of_Events = agg2.Perc_of_Events.round(3)
        except Exception as e:
            print(e)
        try:
            agg2.Gain = agg2.Gain.round(3)
        except Exception as e:
            print(e)
        try:
            cols = ['min_score','max_scr','DepVar','Non_Event','total','Cumulative_Sum','Perc_of_Events','Gain','Bucket']
        except Exception as e:
            print(e)
        try:
            df2 = pd.DataFrame(columns=cols, index=range(1))
        except Exception as e:
            print(e)
        try:
            for a in range(1):
                df2.loc[a].min_score = 0
                df2.loc[a].max_scr = 0
                df2.loc[a].DepVar = 0
                df2.loc[a].Non_Event = 0
                df2.loc[a].total = 0
                df2.loc[a].Cumulative_Sum = 0
                df2.loc[a].Perc_of_Events = 0
                df2.loc[a].Gain = 0
                df2.loc[a].Bucket= 0
        except Exception as e:
            print(e)
        try:
            frames=[df2,agg2]
        except Exception as e:
            print(e)
        try:
            result = pd.concat(frames,ignore_index=True)
        except Exception as e:
            print(e)
            print("Could not generate gains table. Check if function parameters are correct and previous cells are executed")
        try:
            figgains, ax1 = plt.subplots()
        except Exception as e:
            print(e)
        try:
            ax2 = ax1.twinx()
        except Exception as e:
            print(e)
        try:
            line_up, = ax1.plot(result['Gain'], 'o-',color="#ED7038",linewidth=2,label="Gain")
        except Exception as e:
            print(e)
        try:
            line_down, = ax2.plot(result['Bucket'],'k--',linewidth=2,label="Random")
        except Exception as e:
            print(e)
        try:
            ax1.grid(b=True, which='both', color='0.65',linestyle='-')
        except Exception as e:
            print(e)
        try:
            ax1.set_xlabel('Buckets')
        except Exception as e:
            print(e)
        try:
            ax1.set_ylabel("Gain with model")
        except Exception as e:
            print(e)
        try:
            ax2.set_ylabel('Random gain')
        except Exception as e:
            print(e)
        try:
            plt.title('Gains Chart')
        except Exception as e:
            print(e)
        try:
            plt.legend(handles=[line_up, line_down],loc = "lower right")
        except Exception as e:
            print(e)
        try:
            ax2.set_yticklabels(["0","0","20","40","60","80","100"])
        except Exception as e:
            print(e)
        try:
            warnings.filterwarnings('ignore')
        except Exception as e:
            print(e)
        try:
            return(result,figgains)
        except Exception as e:
            print(e)
            print("Error: Could not generate gains Plot. Check if function parameters are correct and previous cells are executed")


    # -----------------------------------function to get KS-----------------------------------
    def getKS(test,dep_var):
        import warnings
        import matplotlib.pyplot as plt
        """
    getKS - A function to get KS-value and plot KS in Python
    =======================================================================================

    **getKS** is a Python function by MathMarket (Team Tesla) of TheMathCompany,
    It aims to create KS-value and plot KS.

    This function returns KS-value at index 0 and plot at index 1

    The module is built using matplotlib,pandas.
    Therefore, these libraries are need to be installed in order to use the module.

    The module consist of one function:
    `getKS(test dataset, dependent variable)`
    """    
        try:
            warnings.filterwarnings('ignore')
        except Exception as e:
            print(e)
        try:
            ks = test.sort_values(by='Pred_Prob',ascending=False)
        except Exception as e:
            print(e)
            print("Could not sort dataframe according to predicted probability")
        try:
            ks['bucket'] = pd.qcut(test.Pred_Prob, 10,duplicates='drop')
        except Exception as e:
            print(e)
        try:
            ks['rank'] = ks['Pred_Prob'].rank(method='first')
        except Exception as e:
            print(e)
        try:
            ks['bucket'] = pd.qcut(ks['rank'].values, 10).codes
        except Exception as e:
            print(e)
        try:
            ks["Event"]=test[dep_var]
        except Exception as e:
            print(e)
        try:
            ks['NonEvent']=np.where(ks[dep_var] == 0 , 1,0)
        except Exception as e:
            print(e)
        try:
            grouped = ks.groupby('bucket', as_index = True)
        except Exception as e:
            print(e)
        try:
            ks_table1 = pd.DataFrame()
        except Exception as e:
            print(e)
        try:    
            ks_table1['min_score']=grouped.min().Pred_Prob
        except Exception as e:
            print(e)
        try:
            ks_table1['max_scr'] = grouped.max().Pred_Prob
        except Exception as e:
            print(e)
        try:
            ks_table1['Event'] = grouped.sum().Event
        except Exception as e:
            print(e)
        try:
            ks_table1['NonEvent'] = grouped.sum().NonEvent
        except Exception as e:
            print(e)
        try:
            ks_table1['total'] = ks_table1.Event + ks_table1.NonEvent
        except Exception as e:
            print(e)
        try:
            ks_table2=ks_table1.iloc[: : -1]
        except Exception as e:
            print(e)
        try:
            pd.options.mode.chained_assignment = None
        except Exception as e:
            print(e)
        try:
            ks_table2.is_copy
        except Exception as e:
            print(e)
        try:
            ks_table2['Cumulative_Sum_Event']=ks_table2['Event'].cumsum()
        except Exception as e:
            print(e)
        try:
            ks_table2['Perc_of_Events']= (ks_table2.Event/ks_table2.Event.sum()) * 100
        except Exception as e:
            print(e)
        try:
            ks_table2['Cumulative_Percent_Event']=ks_table2['Perc_of_Events'].cumsum()
        except Exception as e:
            print(e)
        try:
            ks_table2['Cumulative_Sum_NonEvent']=ks_table2['NonEvent'].cumsum()
        except Exception as e:
            print(e)
        try:
            ks_table2['Perc_of_NonEvents']= (ks_table2.NonEvent/ks_table2.NonEvent.sum()) * 100
        except Exception as e:
            print(e)
        try:
            ks_table2['Cumulative_Percent_NonEvent']=ks_table2['Perc_of_NonEvents'].cumsum()
        except Exception as e:
            print(e)
        try:
            ks_table2['ks'] = np.round(ks_table2.Cumulative_Percent_Event-ks_table2.Cumulative_Percent_NonEvent)
        except Exception as e:
            print(e)
        try:
            cols = ['min_score','max_scr','Event','Non_Event','total','Cumulative_Sum_Event','Perc_of_Events','Cumulative_Percent_Event','Cumulative_Sum_NonEvent','Perc_of_NonEvents','Cumulative_Percent_NonEvent','ks']
        except Exception as e:
            print(e)
        try:
            df2 = pd.DataFrame(columns=cols, index=range(1))
        except Exception as e:
            print(e)
        try:
            for a in range(1):
                df2.loc[a].min_score = 0
                df2.loc[a].max_scr = 0
                df2.loc[a].DepVar = 0
                df2.loc[a].Non_Event = 0
                df2.loc[a].total = 0
                df2.loc[a].Cumulative_Sum_Event = 0
                df2.loc[a].Perc_of_Events = 0
                df2.loc[a].Cumulative_Percent_Event = 0
                df2.loc[a].Cumulative_Sum_NonEvent= 0
                df2.loc[a].Perc_of_NonEvents= 0 
                df2.loc[a].Cumulative_Percent_NonEvent= 0
                df2.loc[a].ks = 0
        except Exception as e:
            print(e)
        try:
            frames=[df2,ks_table2]  
        except Exception as e:
            print(e)
        try:
            result = pd.concat(frames,ignore_index=True)
        except Exception as e:
            print(e)
        try:
            ks_table = pd.DataFrame(ks_table2)
        except Exception as e:
            print(e)
        try:
            ks_table["Bucket"]=[1,2,3,4,5,6,7,8,9,10]
        except Exception as e:
            print(e)
        try:
            ks_table = ks_table.set_index('Bucket')
        except Exception as e:
            print(e)
        try:
            ksvalue = ks_table.ks.max()
        except Exception as e:
            print(e)
        try:
            ksvalue = round(ksvalue,2)
        except Exception as e:
            print(e)
            print("Error: Could not generate Ks table. Check if function parameters are corNon_Eventt and previous cells are executed")
        try:
            figks, ax1 = plt.subplots()
        except Exception as e:
            print(e)
        try:
            ax2 = ax1.twinx()
        except Exception as e:
            print(e)
        try:
            line_down, = ax2.plot(result['Cumulative_Percent_NonEvent'],'k-',linewidth=2,label="Non event")
        except Exception as e:
            print(e)
        try:
            line_up, = ax1.plot(result['Cumulative_Percent_Event'], 'g-',color="#ED7038",linewidth=2,label="Event")
        except Exception as e:
            print(e)
        try:    
            ax1.grid(b=True, which='both', color='0.65',linestyle='-')
        except Exception as e:
            print(e)
        try:
            ax1.set_xlabel('Buckets')
        except Exception as e:
            print(e)
        try:
            ax1.set_ylabel("Event")
        except Exception as e:
            print(e)
        try:
            ax2.set_ylabel('Non Event')
        except Exception as e:
            print(e)
        try:
            plt.title('K-S Chart')
        except Exception as e:
            print(e)
        try:
            ksvalue = str(ksvalue)
        except Exception as e:
            print(e)
        try:
            bbox = dict(boxstyle="round", fc="white")
        except Exception as e:
            print(e)
        try:
            plt.annotate("KS: "+ksvalue, xy=(9.6,22), ha='center', va='center',bbox=bbox)
        except Exception as e:
            print(e)
        try:
            plt.legend(handles=[line_up, line_down],loc = "lower right")
        except Exception as e:
            print(e)
        try:
            ax2.set_yticklabels([])
        except Exception as e:
            print(e)
        try:
            flag = lambda x: '<----(max)' if x == result.ks.max() else ''
        except Exception as e:
            print(e)
        try:
            ks_table['max_ks'] = result.ks.apply(flag)
        except Exception as e:
            print(e)
        try:
            print(ks_table)
        except Exception as e:
            print(e)
        try:
            warnings.filterwarnings('ignore')
        except Exception as e:
            print(e)
        try:
            return(ks_table,figks,{'KS_Value':ksvalue})
        except Exception as e:
            print(e)
            print("Error: Could not generate KS. Check if function parameters are corNon_Eventt and previous cells are executed")

    #----------------------------- report function ---------------------
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    # -------------------------- function to get concordance -----------------------
    def getConcordance(test,dep_var):
        from bisect import bisect_left,bisect_right
        """
    getConcordance - A function to create concordance, discordance, tie pairs in Python
    =========================================================================================

    **getConcordance** is a Python function by MathMarket (Team Tesla) of TheMathCompany,

    The module consist of one function:
    `getConcordance(test dataset, dependent variable)`
    """    
        try:
            zeros = test[(test[dep_var]==0)].reset_index().drop(['index'], axis = 1)
        except Exception as e:
            print(e)
        try:
            ones = test[(test[dep_var]==1)].reset_index().drop(['index'], axis = 1)
        except Exception as e:
            print(e)
        try:
            pred_prob = "Pred_Prob"
        except Exception as e:
            print(e)
        try:
            zeros2 = zeros[[dep_var,pred_prob]].copy()
        except Exception as e:
            print(e)
        try:
            ones2=ones[[dep_var,pred_prob]].copy()
        except Exception as e:
            print(e)
        try:
            zeros2_list = sorted([zeros2.iloc[j,1] for j in zeros2.index])
        except Exception as e:
            print(e)
        try:
            ones2_list = sorted([ones2.iloc[i,1] for i in ones2.index])
        except Exception as e:
            print(e)
        try:
            zeros2_length = len(zeros2_list)
        except Exception as e:
            print(e)
        try:
            ones2_length = len(ones2_list)
        except Exception as e:
            print(e)
        try:
            conc = 0
        except Exception as e:
            print(e)
        try:
            ties = 0
        except Exception as e:
            print(e)
        try:
            disc = 0
        except Exception as e:
            print(e)
        try:
            for i in zeros2.index:
                cur_disc = bisect_left(ones2_list, zeros2.iloc[i,1])
                cur_ties = bisect_right(ones2_list, zeros2.iloc[i,1]) - cur_disc
                disc += cur_disc
                ties += cur_ties
                conc += ones2_length - cur_ties - cur_disc
        except Exception as e:
            print(e)
        try:
            pairs_tested = zeros2_length * ones2_length
        except Exception as e:
            print(e)
        try:
            concordance = conc/pairs_tested
        except Exception as e:
            print(e)
        try:
            discordance = disc/pairs_tested
        except Exception as e:
            print(e)
        try:
            ties_perc = ties/pairs_tested
        except Exception as e:
            print(e)
        try:
            print("Concordance = ", (concordance*100) , "%")
        except Exception as e:
            print(e)
        try:
            print("Discordance = ", (discordance*100) , "%")
        except Exception as e:
            print(e)
        try:
            print("Tied = ", (ties_perc*100) , "%")
        except Exception as e:
            print(e)
        try:
            print("Pairs = ", pairs_tested)
        except Exception as e:
            print(e)
        try:
            return({"Concordance":concordance,'Discordance':discordance,'Ties':ties_perc})
        except Exception as e:
            print(e)
            print("Error: Check if function parameters are correct and previous cells are executed")


    # ---------------------- to show feature importance --------------------
    def show_features_imp(xgb_model=None,target=target):
        dep_var = target
        feat_imp_gbm=xgb_model.feature_importances_
        train_new1 = train.drop([dep_var], axis = 1)
        feature_gbm=train_new1.columns
        dfnew = train.drop([dep_var], axis = 1)
        feat_gbm=pd.DataFrame(feature_gbm,columns=['Feature'])
        #feat_gbm["       "] = dfnew.apply(lambda _: '>',axis=1)
        feat_importance_gbm=pd.DataFrame(feat_imp_gbm,columns=['Feature Importance'])
        df_feat_imp1_gbm = pd.concat([feat_gbm,feat_importance_gbm], axis=1)
        df_feat_imp_gbm = df_feat_imp1_gbm.to_string(sparsify=bool)
        print(df_feat_imp1_gbm.sort_values('Feature Importance',ascending=False))

        return


    # -----------------------------to create folder skleton ------------------------------
    def create_skleton():
        def get_uniq_model_run_name():
            import time
            uniq_name=f'Model_run_{time.strftime("%d_%b_%Y_%H_%M_%S")}'
            return uniq_name

        model_run_path=os.path.join(os.getcwd(),get_uniq_model_run_name())
        os.mkdir(model_run_path)
        gain_path=os.path.join(model_run_path,'Gain')
        os.mkdir(gain_path)
        ks_path=os.path.join(model_run_path,'KS')
        os.mkdir(ks_path)
        sweetviz_path=os.path.join(model_run_path,'sweetviz_comparisons')
        os.mkdir(sweetviz_path)
        model_path=os.path.join(model_run_path,'Model')
        os.mkdir(model_path)
        file_store_path=os.path.join(model_run_path,'Data')
        os.mkdir(file_store_path)


        return gain_path,ks_path,sweetviz_path,model_path,file_store_path

    # 
    #  # importing required libriries
    logger = get_logger('model_managment')
    logger.info('Importing required libraries')
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    import logging

    from xgboost import XGBClassifier
    from  sklearn.model_selection import RandomizedSearchCV

    import os

    import pickle

    import  sweetviz as sv

    # ------------------------ setting index -------------------

    logger.info('setting index started')
    train.set_index(index_col,drop=True,inplace=True)


    logger.info('splitting x y of train')
    # splitting  train into x,y
    x_train,y_train=train.drop(target,axis=1),train[target]

    if parameter_optimization:
        
        logger.info('Hyperparameter tuning started')
        xgb=XGBClassifier(objective='binary:logistic')
        random_search=RandomizedSearchCV(xgb,n_jobs=-1,cv=CV,n_iter=n_iter,scoring=scoring,
                                        param_distributions=param_grid)

        random_search.fit(x_train,y_train)
        print(report(random_search.cv_results_,show_n_report))
        logger.info('Hyper parameter tuning completed')


    logger.info(' model defining started')
    if best_model and parameter_optimization:
        model=random_search.best_estimator_
    else:
        logger .info('Model defining without hyper parameter tuning')
        from xgboost import XGBClassifier
        model=XGBClassifier(objective=objective,subsample=subsample,scale_pos_weight=scale_pos_weight,reg_lambda=reg_lambda,        reg_alpha=reg_alpha,n_estimators=n_estimators,
                   min_child_weight=min_child_weight, max_depth=max_depth, max_delta_step=max_delta_step, learning_rate=learning_rate, gamma=gamma,
                   colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel)

        model.fit(x_train,y_train)


    if show_feature_importance:
        logger.info('Feature importance showing')
        show_features_imp(xgb_model=model,target=target)
    # ------------------- creating skleton---------------------
    logger.info('Required folder creation started ')
    gain_path,ks_path,sweetviz_path,model_path,file_store_path=create_skleton()

    # ----------------------saving model ----------------------
    logger.info('model saving started')
    
    filename = model_name+'.pkl'
    pickle.dump(model, open(os.path.join(model_path,filename), 'wb'))
    logger.info('model saving done')

        

    # ------------------------------ predictions --------------------------------
    print('--------------------train performance ----------------')
    logger.info('Prediction for train started')
    train["Pred_Prob"]=model.predict_proba(x_train)[:,1]
    train.to_csv(os.path.join(file_store_path,'train_xgb.csv'))
    logger.info('train file saved')
    # getConcordance(train,target)
       #----------------------- for train 
    # -- gain
    logger.info('Gain for train month started')
    gains_train = createGains(train,target)
    gains_table_train=gains_train[0]
    print('--------- Train gain-------')
    print(gains_table_train) 
    gains_table_train.to_csv(os.path.join(gain_path,'train_GAIN.csv'))

    #-- ks
    logger.info('KS for train month started')
    oot_apr_log1 = train.reset_index(drop=True)
    ks_rf_oot_apr = getKS(oot_apr_log1,target)
    ks_rf_oot_apr[0].to_csv(os.path.join(ks_path,'train_KS.csv'))
    
    # concordance
    


    print('performance and prediction for tetst')
    if tests:
        for i in range(len(tests)):
            
            # print('im here')
            test_df=tests[i]
            test_df_name=test_file_names[i]
            logger.info(f'{test_df_name} report generation started')
            print(f'<<<<<<<<<<<<<<<<-------{test_df_name}------->>>>>>>>>>>>>>')
            # print(test_df.columns)
            test_df.set_index(index_col,drop=True,inplace=True)
            logger.info(f'prediction for {test_df_name} ')
            test_df["Pred_Prob"]=model.predict_proba(test_df.drop(target,axis=1))[:,1]
            print(os.path.join(file_store_path,f'{test_df_name}_xgb.csv'))
            test_df.to_csv(os.path.join(file_store_path,f'{test_df_name}_xgb.csv'))
            logger.info(f'{test_df_name} file saved')
            # print('prediction_done')
            logger.info(f' gain for {test_df_name} started')
            gains_train = createGains(test_df,target)
            gains_table_train=gains_train[0]
            print('--------- Train gain-------')
            print(gains_table_train) 
            gain_file_name=f'{test_df_name}_GAIN.csv'
            gains_table_train.to_csv(os.path.join(gain_path,gain_file_name))
            logger.info(f' gain for {test_df_name} saved')
            #-- ks
            logger.info(f' KS for {test_df_name} started')
            oot_apr_log1 = test_df.reset_index(drop=True)
            ks_rf_oot_apr = getKS(oot_apr_log1,target)
            ks_file_name=f'{test_df_name}_KS.csv'
            ks_rf_oot_apr[0].to_csv(os.path.join(ks_path,ks_file_name))
            logger.info(f' KS for {test_df_name} file saved')

            # concordance discordance
            # getConcordance(test_df,target)


            # sweetviz comparison
            logger.info(f'sweetviz comparison for train and {test_df_name} started')
            
            compare_report1 = sv.compare([train, 'Train'], [test_df, test_df_name])
            sweetviz_file_name=f'train_{test_df_name} compare.html'
            compare_report1.show_html(os.path.join(sweetviz_path,sweetviz_file_name), open_browser=False)
            logger.info(f'sweetviz comparison for train and {test_df_name} file saved')

    return 