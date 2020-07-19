import numpy as np
import pandas as pd
import scipy 
# import boruta
# from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.utils.fixes import loguniform


class CCARD_MODEL(): 
    
    """
        This class implements CCARD level model with feature selection
        
        Parameters
        ----------
        feature_selection : bool
            Perform feature selection or not
        hyperparameter_tuning : bool
            Tune the model or not
        model_name : str
            Name of the model (default is ridge)
            Support 'ridge', 'SVR' in backcast function
        transformation: str
            'log' or 'sqrt' transformation of y
        random_state: int
            set seed for the model 
    
    """
    
    # some class variables (hard-coded)
    LASSO_FEATURE_SELECTION_PENALTY = 0.005
    RIDGE_PENALTY = 0.001 # default choice based on tuning before 2019-11-03
    SVR_C, SVR_epislon = 10.0, 0.2 # from Vicky's model 
    EPSILON_LOG_ACTUAL = 0.001
    
    
    def __init__(self, feature_selection = False, hyperparameter_tuning = True, training_to = None,  model_name = 'ridge', transformation = 'log', random_state=0): 
        
        self.random_state = random_state
        self.training_to = training_to
        self.feature_selection = feature_selection
        self.model_name = model_name
        self.transformation = transformation
        self.selected_cols = None
        self.hyperparameter_tuning = hyperparameter_tuning


    # some helper functions   
    @staticmethod 
    def remove_collinearity(df, r_thresh = 0.99): 
        """
            identify collinear features
        """
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > r_thresh)]
        return to_drop

            
    @staticmethod
    def _remove_extra_vars(X): 
        """
            remove features before training
        """
        removed_vars = ['log_actual', 'date',  'log_alp', 'sqrt_actual', 'asin','Customer-Facing Name', 'log_baseline', 'holiday', 'program', 'country', 'actual', 'device_type', 'qty', 'asin_start_date', 'dtcp']
        for var in removed_vars:
            if var in X.columns: 
                X = X.drop(var, axis = 1)
        
        return X
    
    @staticmethod
    def compute_mape_bias(y_true, y_pred, clip = True):
        """
            compute MAPE and bias 
        """
        bias = (np.sum(y_pred) - np.sum(y_true))/np.sum(y_true)
        pe = np.abs(bias)
        rmse = np.sqrt(np.mean(pow(y_pred-y_true,2)))

        if clip: 
            bias = np.clip(bias,-1,1)
            pe = min(pe,1)

        return pe, bias, rmse
        

    def _perform_stability_selection(self, alphas, X, y, n_bootstrap_iterations = 100, seed = 0):
        """
            perform bootstrapped feature selection (pre-alpha mode)
        """
    
        n_samples, n_variables = X.shape
        n_alphas = alphas.shape[0]
        rnd = np.random.RandomState(seed)
        selected_variables = np.zeros((n_variables,n_bootstrap_iterations))
        stability_scores = np.zeros((n_variables, n_alphas))

        for idx, alpha, in enumerate(alphas):
            for iteration in range(n_bootstrap_iterations):
                bootstrap = rnd.choice(np.arange(n_samples),
                                     size= int(n_samples*0.8),
                                     replace=False)

                X_train = X.iloc[bootstrap,:]
                y_train = y.iloc[bootstrap]

                params = {'alpha': alpha}
                lasso = Lasso(**params)
                lasso.fit(X_train, y_train)
                selected_variables[:, iteration] = (np.abs(lasso.coef_) > 1e-4)

            stability_scores[:, idx] = selected_variables.mean(axis=1)

        self.selected_cols = X_train.columns[stability_scores[:,0] > .25].values

        
    def _perform_feature_selection(self, X, y, algo = 'lasso'):
        """
            perform feature selection using 'lasso' or 'boruta' (boruta is not supported by Eider yet)
        """
        
        if self.selected_cols is None: 
            X = self._remove_extra_vars(X) # remove extra vars from X
            
            if algo == 'boruta':
                rf_boruta = RandomForestRegressor(n_jobs=40, random_state=self.random_state)
                boruta = BorutaPy(rf_boruta, n_estimators=100, verbose=2, alpha = 0.05)
                boruta.fit(X.values, y.values.ravel())
                self.selected_cols = X.columns[boruta.support_]
            elif algo == 'lasso': 
                lr = Lasso(alpha = self.LASSO_FEATURE_SELECTION_PENALTY).fit(X, y)
                model = SelectFromModel(lr, prefit=True)
                self.selected_cols = np.unique(np.concatenate([X.columns[model.get_support()].values, ['covid', 'asp']]))
#                 print(self.selected_cols)
            else: 
                print('currently supporting only Boruta or Lasso')

                   
    def perform_hyperparameter_tuning(self, X,y, model_name = 'ridge', n_values = 100):
        if model_name == 'ridge':
#             model = Ridge()
#             reg_pipeline = Pipeline([('scaler', MinMaxScaler()), 
#                             ('Ridge', Ridge())])

#             param_grid = [{'alpha': np.logspace(-5,5,100)}]
            param_dist = {'alpha': loguniform(1e-5, 1e0)}
            clf = RandomizedSearchCV(estimator = Ridge(normalize = True), param_distributions = param_dist, n_iter = 50, n_jobs = 10, random_state = self.random_state)
            clf.fit(X, y)
            return clf.best_params_ 
            
        else: 
            print("Only supporting Ridge for now")
        
    def fit(self, X,y):
        
        #  length checks
        assert(X.shape[0] == len(y))
        X = self._remove_extra_vars(X)
        X.replace([np.inf, -np.inf], np.nan,inplace=True)
        X.fillna(X.mean(), inplace = True) # fill nan's values    


        if self.feature_selection and (self.selected_cols is None):
            print('running feature selection ...')
            self._perform_feature_selection(X,y)
        
        X = X[self.selected_cols]
    
        
        if self.model_name == 'ridge':
                
            best_params = {'alpha':self.RIDGE_PENALTY} # default hyperparameters
            if self.hyperparameter_tuning: 
                print("Running hyperparameter tuning ...")
                best_params = self.perform_hyperparameter_tuning(X, y, self.model_name)

            # print('The best param(s) are {}'.format(best_params))
            self.model = Ridge(**best_params)

        self.model.fit(X, y)
            
    def predict(self, X): 
        
        X = self._remove_extra_vars(X)
        X.replace([np.inf, -np.inf], np.nan,inplace=True)
        X.fillna(X.mean(), inplace = True) # fill nan's values    
        y_pred = self.model.predict(X)
        vlt_vars = X.columns[['vlt' in x for x in X.columns]]
        
        for vlt_var in vlt_vars: # set VLT vars to 0
            X[vlt_var] = 0.0 

        y_pred = self.model.predict(X)
        if self.transformation == 'log':
            y_pred = np.exp(y_pred)
        if self.transformation == 'sqrt': 
            y_pred = pow(y_pred,2)
            
        return y_pred 
        

    # perform backcast on certain periods
    def back_cast(self, df, backcast_periods, END_DATE, horizon, metric_clip = True, retrain = True): 
        """
            Parameters
            ----------
            retrain : bool
                Redo feature selection for each new forecast version 
                
        """
        
        df.replace([np.inf, -np.inf], np.nan,inplace=True)

                
        result_vec = []
        for i, training_to in enumerate(backcast_periods): 
            print('Fold {}'.format(i+1))
            df_result = self.evaluate_model(df, training_to, END_DATE, horizon, metric_clip, model_name = self.model_name, retrain = retrain)
            if df_result is not None:
                result_vec.append(df_result)
        
        df_result_all = pd.concat(result_vec)
#         print(df_result_all)
        
        # save program-level result
        self.program_result = df_result_all
        
        # get program level MAPEs and Biases
        weight_vec, wmape_vec, wbias_vec = [], [], []
        programs = df_result_all.program.unique()
        program_vec = []
        for program in programs:
            temp = df_result_all[df_result_all.program == program]
            wmape = np.sum(temp['ape']*temp['weight'])/np.sum(temp['weight'])
            wbias = np.sum(temp['bias']*temp['weight'])/np.sum(temp['weight'])
            weight = np.sum(temp['weight'])
            wmape_vec.append(wmape)
            wbias_vec.append(wbias)
            program_vec.append(program)
            weight_vec.append(weight)

        df_horizon = pd.DataFrame({'program': program_vec, 'wmape': wmape_vec, 'wbias': wbias_vec, 'weight':weight_vec})
        df_horizon['horizon'] = horizon
        df_horizon = df_horizon.sort_values(by = ['program'])
        
        # get topline level MAPE and BIAS
        w_vec, b_vec, wei_vec = [], [], []
        for training_to in df_result_all.training_to.unique():
            weight_vec, wmape_vec, wbias_vec = [], [], []
            programs = df_result_all.program.unique()
            program_vec = []
            df_temp = df_result_all[df_result_all.training_to == training_to]
            for program in df_temp.program.unique():
                temp = df_temp[df_temp.program == program]
                wmape = np.sum(temp['ape']*temp['weight'])/np.sum(temp['weight'])
                wbias = np.sum(temp['bias']*temp['weight'])/np.sum(temp['weight'])
                weight = np.sum(temp['weight'])
                wmape_vec.append(wmape)
                wbias_vec.append(wbias)
                program_vec.append(program)
                weight_vec.append(weight)

            wmape_vec = np.array(wmape_vec)
            wbias_vec = np.array(wbias_vec)
            program_vec = np.array(program_vec)
            wmape = np.nansum(wmape_vec*weight_vec)/np.nansum(weight_vec)
            wbias = np.nansum(wbias_vec*weight_vec)/np.nansum(weight_vec)
            weight = np.nansum(weight_vec)

            w_vec.append(wmape)
            b_vec.append(wbias)
            wei_vec.append(weight)
            
        w_vec, b_vec, wei_vec = np.array(w_vec), np.array(b_vec), np.array(wei_vec)
        
        # topline mape bias
        wmape_aucc, wbias_aucc = np.sum(w_vec*wei_vec)/np.sum(wei_vec), np.sum(b_vec*wei_vec)/np.sum(wei_vec)

        df_horizon = df_horizon.append({'program': 'AuCC', 'wmape':wmape_aucc, 'wbias': wbias_aucc, 'horizon':horizon}, ignore_index = True)
        
        return df_horizon, df_result_all
    
    
    def evaluate_model(self, df, training_to, END_DATE, horizon, metric_clip, model_name = 'LR', save_forecasts = False, retrain = True): 
        
        print('Preparing training and test sets ...')
        
        df_program = df[['date', 'program']]
        
        if (training_to + pd.Timedelta(days = horizon*7-1)) < pd.to_datetime(END_DATE): 


            forecast_to = training_to + pd.Timedelta(days = horizon*7)
            forecast_to = forecast_to.strftime('%Y-%m-%d')
            training_to = training_to.strftime('%Y-%m-%d')
            print('Training model up to {} and forecasting up to {}'.format(training_to, forecast_to))

        
            if model_name == 'SVR': # create trends for SVR 
                df['trend_6m'] = (pd.to_datetime(df['date'])-(pd.to_datetime(training_to) - pd.Timedelta(30*6))).dt.days
                df['trend_6m'] = np.where(df['trend_6m']<=0,0,np.log(df['trend_6m']))
                df['trend_12m'] = (pd.to_datetime(df['date'])-(pd.to_datetime(training_to) - pd.Timedelta(30*12))).dt.days
                df['trend_12m'] = np.where(df['trend_12m']<=0,0,np.log(df['trend_12m']))
            

            # create train/test sets
            train = df[df.date < training_to]
            test = df[(df.date >= training_to) & (df.date < forecast_to)]
            if self.transformation == 'log':
                y_train, X_train = np.log(train['actual'] + self.EPSILON_LOG_ACTUAL), self._remove_extra_vars(train)
                y_test, X_test = np.log(test['actual'] + self.EPSILON_LOG_ACTUAL), self._remove_extra_vars(test)
                
            if self.transformation == 'sqrt':
                y_train, X_train = np.sqrt(train['actual']), self._remove_extra_vars(train)
                y_test, X_test = np.sqrt(test['actual']), self._remove_extra_vars(test)
    
            X_train.fillna(X_train.mean(), inplace = True) # fill nan's values    
            X_test.fillna(X_test.mean(), inplace = True) # fill nan's values
            
            # reset feature set for each forecast version
            if retrain: 
                self.selected_cols = None

        
            # feature selection
            if self.selected_cols is None: 
                print('Running feature selection ...')
                self._perform_feature_selection(X_train, y_train)

                    
            X_train = X_train[self.selected_cols]
            X_test = X_test[self.selected_cols]
            
            
#             to_drop = self.remove_collinearity(X_train, r_thresh = 0.80)
#             print('collinear features: {}'.format(to_drop))
#             X_train = X_train.drop(to_drop, axis = 1)
#             X_test = X_test.drop(to_drop, axis = 1)
            
            n_unique = X_train.apply(lambda x: len(np.unique(x))) 
            selected_vars = X_train.columns[~((n_unique > 2) & (n_unique < 40))]
            var_list = X_train[selected_vars].columns[X_train[selected_vars].apply(lambda x: np.std(x)) == 0]
            print('Dropping constant vars: {}'.format(var_list.values))
            X_train = X_train.drop(var_list, axis = 1)# test correlations
            X_test = X_test.drop(var_list, axis = 1)
            train = train.drop(var_list, axis = 1)

            train_program = df_program[df_program.date < training_to]
            test_program = df_program[(df_program.date >= training_to) & (df_program.date < forecast_to)]
 

            # train model and generate forecasts
            if model_name == 'SVR': # based on Vicky's code
                sc_y = StandardScaler()
                y_pre = train[['sqrt_actual']]
                y = sc_y.fit_transform(y_pre)
                X_pre = X_train
                pipeline = Pipeline([('scaler', StandardScaler()),('estimator', SVR(C=self.SVR_C, epsilon=self.SVR_epsilon, gamma='auto', kernel='poly',degree = 3))])
                pipeline.fit(X_pre, y)
                X_test['instock_pct'] = 1.0 
                y_pred = sc_y.inverse_transform(pipeline.predict(X_test))

            if model_name == 'LR':
        
                model = LinearRegression()
                print(X_train.columns.values)
                model.fit(X_train,y_train)
                print('max coef. {}'.format(np.max(np.abs(model.coef_)))) # check coefs
                vlt_vars = X_test.columns[['vlt' in x for x in X_test.columns]]
                for vlt_var in vlt_vars: # set VLT vars to 0
                    X_test['vlt'] = 0.0 # unconstrained forecast
                
                y_pred = model.predict(X_test)
    
            if model_name == 'ridge':
                
                best_params = {'alpha':self.RIDGE_PENALTY} # default hyperparameters
                if self.hyperparameter_tuning: 
                    print("Running hyperparameter tuning ...")
                    best_params = self.perform_hyperparameter_tuning(X_train, y_train, self.model_name)
                
                print('The best param(s) are {}'.format(best_params))
                model = Ridge(**best_params, normalize = True)
                model.fit(X_train,y_train)
                
                vlt_vars = X_test.columns[['vlt' in x for x in X_test.columns]]
                for vlt_var in vlt_vars: # set VLT vars to 0
                    X_test[vlt_var] = 0.0 
                y_pred = model.predict(X_test)
    
            if model_name == 'KRR': # need to reoptimize later
                params = {'alpha': 0.0001,'gamma': 0.0022,'kernel': 'laplacian'}
                model = KernelRidge(**params)
                reg_pipeline = make_pipeline(preprocessing.StandardScaler(), model)
                reg_pipeline.fit(X_train,y_train)
                X_test['vlt'] = 0.0 # unconstrained forecast
                y_pred = reg_pipeline.predict(X_test)


            if save_forecasts: 
                test['y_pred'] = y_pred
                self.forecasts = test

            # evaluate performance at program-level
            ape_vec = []
            weight_vec = []
            bias_vec = []
            training_to_vec = []
            program_vec = []
            for program in train_program.program.unique():
                mask = (test_program.program == program).values
                if np.sum(mask) > 0:
                    y_pred_program = y_pred[mask]
                    y_test_program = y_test[mask]
                    
                    # check lengths
                    assert len(y_pred_program) == len(y_test_program)
                    
                    # untransform y
                    if self.transformation == 'log': 
                        y_test_program = np.exp(y_test_program) - self.EPSILON_LOG_ACTUAL
                        y_pred_program = np.exp(y_pred_program) - self.EPSILON_LOG_ACTUAL
                    elif self.transformation == 'sqrt':
                        y_test_program = pow(y_test_program,2)
                        y_pred_program = pow(y_pred_program,2)

                    ape, bias, rmse = self.compute_mape_bias(y_test_program, y_pred_program, metric_clip)
                    weight_vec.append(np.sum(y_test_program))
                    ape_vec.append(ape)
                    bias_vec.append(bias)
                    program_vec.append(program)

            df_result = pd.DataFrame({'program':program_vec, 'ape':ape_vec, 'weight':weight_vec, 'bias':bias_vec})
            df_result['training_to'] = training_to
            
#             print(df_result)
            return df_result
        else:
            print('Less than {} weeks to forecast'.format(horizon))
            return None
    
    

    